import torch
import torch.nn as nn
import logging
import gc
from copy import deepcopy

from ultralytics.nn.tasks import DetectionModel
import ultralytics.nn.tasks as tasks

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SE(nn.Module):
    def __init__(self, c1, c2=None, *args, **kwargs):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c1, c1 // 16 if c1 >= 16 else 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c1 // 16 if c1 >= 16 else 1, c1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class CBAM(nn.Module):
    def __init__(self, c1, c2=None, kernel_size=7, *args, **kwargs):
        super().__init__()
        self.channel_attention = SE(c1)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.channel_attention(x)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.spatial_attention(torch.cat([max_out, avg_out], dim=1))
        return x * spatial_out

tasks.SE = SE
tasks.CBAM = CBAM

class GenomeDecoder:
    """
    1D Genome -> In-Memory YOLO Dict 변환 및 파싱
    """
    BLOCK_MAP = {0: 'C2f', 1: 'C3k2', 2: 'GhostConv'}
    ATTN_MAP = {0: None, 1: 'CBAM', 2: 'SE'}
    MUTABLE_INDICES = [4, 6, 8, 12] 
    
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.base_cfg = {'nc': self.num_classes, 'scales': {'n': [0.33, 0.25, 1024]}}
        self.original_split_idx = 10 
        
        self.base_layers = [
            [-1, 1, 'Conv', [64, 3, 2]],   
            [-1, 1, 'Conv', [128, 3, 2]],  
            [-1, 3, 'C2f', [128, True]],   
            [-1, 1, 'Conv', [256, 3, 2]],  
            [-1, 6, 'C2f', [256, True]],   
            [-1, 1, 'Conv', [512, 3, 2]],  
            [-1, 6, 'C2f', [512, True], 'tag_P3'],   
            [-1, 1, 'Conv', [1024, 3, 2]], 
            [-1, 3, 'C2f', [1024, True], 'tag_P4'],  
            [-1, 1, 'SPPF', [1024, 5]],    
            [-1, 1, 'nn.Upsample', [None, 2, 'nearest']], 
            [[-1, 6], 1, 'Concat', [1]],                  
            [-1, 3, 'C2f', [512], 'tag_P5'],              
            [[15, 18, 21], 1, 'Detect', [self.num_classes]] 
        ]

    def _convert_to_absolute_indices(self, layers):
        for i, layer in enumerate(layers):
            f_idx = layer[0]
            if isinstance(f_idx, int):
                layers[i][0] = i + f_idx if f_idx < 0 else f_idx
            elif isinstance(f_idx, list):
                layers[i][0] = [i + x if x < 0 else x for x in f_idx]
        return layers

    def decode(self, genome: list):
        expected_len = len(self.MUTABLE_INDICES) * 3
        if len(genome) != expected_len:
            raise ValueError(f"Genome length validation failed")

        layers = deepcopy(self.base_layers)
        layers = self._convert_to_absolute_indices(layers)
        
        self.layer_map = {i: i for i in range(len(layers))} 
        insertions_before_split = 0
        inserted_attention_indices = [] 
        
        for i, target_idx in reversed(list(enumerate(self.MUTABLE_INDICES))):
            block_code, depth, attn_code = genome[i*3:(i+1)*3]
            
            layers[target_idx][1] = depth
            layers[target_idx][2] = self.BLOCK_MAP[block_code]
            
            if self.ATTN_MAP[attn_code] is not None:
                insert_idx = target_idx + 1
                base_channels = layers[target_idx][3][0] 
                
                # YOLO11n width_multiple(0.25) 스케일링 적용 (Shape Mismatch 방지)
                scaled_channels = int(base_channels * 0.25)
                
                attn_layer = [target_idx, 1, self.ATTN_MAP[attn_code], [scaled_channels]]
                
                if len(layers[target_idx]) == 5:
                    tag = layers[target_idx].pop()
                    attn_layer.append(tag)

                layers.insert(insert_idx, attn_layer)
                inserted_attention_indices.append(insert_idx)
                
                if insert_idx <= self.original_split_idx:
                    insertions_before_split += 1
                
                new_map = {}
                for old_i, new_i in self.layer_map.items():
                    new_map[old_i] = new_i + 1 if new_i >= insert_idx else new_i
                self.layer_map = new_map
                
                for j in range(insert_idx + 1, len(layers)):
                    f_idx = layers[j][0]
                    if isinstance(f_idx, int) and f_idx >= insert_idx:
                        layers[j][0] = f_idx + 1
                    elif isinstance(f_idx, list):
                        layers[j][0] = [x + 1 if (isinstance(x, int) and x >= insert_idx) else x for x in f_idx]

        # Detect Head 동적 라우팅
        tag_to_idx = {}
        for i, layer in enumerate(layers):
            if len(layer) == 5 and str(layer[4]).startswith('tag_'):
                tag_to_idx[layer[4]] = i
                layer.pop() 
        
        try:
            detect_inputs = [tag_to_idx['tag_P3'], tag_to_idx['tag_P4'], tag_to_idx['tag_P5']]
            layers[-1][0] = detect_inputs 
        except KeyError as e:
            raise RuntimeError(f"Architecture breakdown: Missing Required Feature Pyramid Tag {e}")

        cfg = deepcopy(self.base_cfg)
        dynamic_split = self.original_split_idx + insertions_before_split
        cfg['backbone'] = layers[:dynamic_split]
        cfg['head'] = layers[dynamic_split:]
        
        return cfg, self.layer_map, inserted_attention_indices

class WeightSurgeon:
    def __init__(self, pretrained_path="yolo11n.pt"):
        ckpt = torch.load(pretrained_path, map_location="cpu")
        model_obj = ckpt.get("ema") or ckpt.get("model")
        self.seed_state_dict = {k: v.clone() for k, v in model_obj.state_dict().items()}
        del ckpt, model_obj
        
    @torch.no_grad()
    def transplant(self, cfg_dict: dict, layer_map: dict) -> DetectionModel:
        new_model = DetectionModel(cfg=cfg_dict)
        new_state_dict = new_model.state_dict()
        
        # Weight/Bias 상속률 계산
        weight_keys = [k for k in new_state_dict.keys() if 'weight' in k or 'bias' in k]
        total_weights = len(weight_keys)
        inherited_weights = 0
        
        for seed_key, seed_tensor in self.seed_state_dict.items():
            parts = seed_key.split('.')
            if parts[0] == 'model' and parts[1].isdigit():
                old_idx = int(parts[1])
                if old_idx in layer_map:
                    parts[1] = str(layer_map[old_idx])
                    new_key = '.'.join(parts)
                    if new_key in new_state_dict and new_state_dict[new_key].shape == seed_tensor.shape:
                        new_state_dict[new_key].copy_(seed_tensor)
                        if new_key in weight_keys:
                            inherited_weights += 1

        new_model.load_state_dict(new_state_dict)
        self._apply_exact_identity_zero_init(new_model)
        self._force_stride_calculation(new_model)
        return new_model

    def _apply_exact_identity_zero_init(self, model: nn.Module):
        for module in model.modules():
            if type(module).__name__ in ['CBAM', 'SE']: 
                prev_layer = None
                for child in module.modules():
                    if isinstance(child, nn.Sigmoid) and prev_layer is not None:
                        nn.init.zeros_(prev_layer.weight)
                        if hasattr(prev_layer, 'bias') and prev_layer.bias is not None:
                            nn.init.constant_(prev_layer.bias, 5.0)
                        prev_layer = None 
                    elif isinstance(child, (nn.Conv2d, nn.Linear)):
                        prev_layer = child

    @torch.no_grad()
    def _force_stride_calculation(self, model: nn.Module):
        model.eval()
        p = next(model.parameters())
        dummy_input = torch.zeros(1, 3, 640, 640, device=p.device, dtype=p.dtype)
        _ = model(dummy_input)
        
    def cleanup(self):
        self.seed_state_dict.clear()
        gc.collect()
        torch.cuda.empty_cache()