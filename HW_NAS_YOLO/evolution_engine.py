import numpy as np
import random
import logging
from typing import List, Dict, Set

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GenomeOptimizer:
    def __init__(self, num_stages=4):
        self.num_stages = num_stages
        self.bounds = {'block': [0, 2], 'depth': [1, 9], 'attn': [0, 2]}

    def generate_random_genome(self) -> List[int]:
        genome = []
        for _ in range(self.num_stages):
            genome.extend([
                random.randint(self.bounds['block'][0], self.bounds['block'][1]),
                random.randint(self.bounds['depth'][0], self.bounds['depth'][1]),
                random.randint(self.bounds['attn'][0], self.bounds['attn'][1])
            ])
        return genome

    def crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        child = []
        for i in range(self.num_stages):
            idx = i * 3
            chosen_parent = parent1 if random.random() < 0.5 else parent2
            child.extend(chosen_parent[idx : idx+3])
        return child

    def mutate(self, genome: List[int], mutation_rate=0.1) -> List[int]:
        mutated = list(genome)
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                gene_type = i % 3
                bound = self.bounds['block'] if gene_type == 0 else (self.bounds['depth'] if gene_type == 1 else self.bounds['attn'])
                mutated[i] = random.randint(bound[0], bound[1])
        return mutated


class NSGA2Engine:
    def __init__(self, pop_size=30, max_gen=30):
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.optimizer = GenomeOptimizer()
        
        self.best_hv_proxy = 0.0 
        self.stagnation_counter = 0
        self.STAGNATION_PATIENCE = 3 
        
        # Hypervolume 평가를 위한 Global Reference Latency
        self.global_ref_latency = None 

    def non_dominated_sort(self, population_data: List[Dict], current_gen: int):
        anneal_end_gen = self.max_gen * 0.4
        w_slope = max(0.0, 1.0 - (current_gen / anneal_end_gen))

        for p in population_data:
            p['effective_slope'] = p['slope'] * w_slope

        def dominates(p, q):
            return (p['mAP'] >= q['mAP'] and p['effective_slope'] >= q['effective_slope'] and p['latency'] <= q['latency']) and \
                   (p['mAP'] > q['mAP'] or p['effective_slope'] > q['effective_slope'] or p['latency'] < q['latency'])

        fronts = [[]]
        for p in population_data:
            p['domination_count'] = 0
            p['dominated_solutions'] = []
            for q in population_data:
                if dominates(p, q):
                    p['dominated_solutions'].append(q)
                elif dominates(q, p):
                    p['domination_count'] += 1
            if p['domination_count'] == 0:
                p['rank'] = 1
                fronts[0].append(p)

        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in p['dominated_solutions']:
                    q['domination_count'] -= 1
                    if q['domination_count'] == 0:
                        q['rank'] = i + 2
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
            
        return fronts[:-1]

    def _calculate_hypervolume_proxy(self, front0: List[Dict]) -> float:
        if not front0: return 0.0
        
        if self.global_ref_latency is None:
            worst_latency = max([p['latency'] for p in front0])
            self.global_ref_latency = worst_latency * 1.2 if worst_latency > 0 else 100.0
            logger.info(f"Locked Global Reference Latency at {self.global_ref_latency:.2f} ms")
        
        hv_area = sum([p['mAP'] * max(0, self.global_ref_latency - p['latency']) for p in front0])
        return hv_area

    def generate_next_generation(self, evaluated_population: List[Dict], current_gen: int, global_history: Set[tuple]) -> List[List[int]]:
        
        if not evaluated_population:
            logger.error("🚨 All models in the previous generation failed! Generating a random fallback population to survive.")
            return [self.optimizer.generate_random_genome() for _ in range(self.pop_size)]

        fronts = self.non_dominated_sort(evaluated_population, current_gen)
        
        current_hv = self._calculate_hypervolume_proxy(fronts[0])
        if current_hv > self.best_hv_proxy * 1.01:
            self.best_hv_proxy = current_hv
            self.stagnation_counter = 0
            logger.info(f"Gen {current_gen}: Pareto Front improved (Global HV: {current_hv:.2f})")
        else:
            self.stagnation_counter += 1
            logger.warning(f"Gen {current_gen}: Stagnation detected ({self.stagnation_counter}/{self.STAGNATION_PATIENCE})")

        survivors = []
        for front in fronts:
            survivors.extend(front)
            if len(survivors) >= self.pop_size // 2:
                break
        survivors = survivors[:self.pop_size // 2]
        
        next_generation_genomes = [s['genome'] for s in survivors]
        
        # 세대 간 중복 평가 방지 로직
        unique_genomes = set([tuple(g) for g in next_generation_genomes])
        unique_genomes.update(global_history)

        immigrant_count = 0
        if self.stagnation_counter >= self.STAGNATION_PATIENCE:
            immigrant_count = max(1, int(self.pop_size * 0.1)) 
            logger.warning(f"🚨 Stagnation Triggered! Injecting {immigrant_count} random immigrants.")
            self.stagnation_counter = 0 

        attempts = 0
        while len(next_generation_genomes) < self.pop_size - immigrant_count and attempts < 1000:
            p1 = random.choice(survivors)['genome']
            p2 = random.choice(survivors)['genome']
            child = self.optimizer.crossover(p1, p2)
            child = self.optimizer.mutate(child)
            
            child_tuple = tuple(child)
            if child_tuple not in unique_genomes:
                unique_genomes.add(child_tuple)
                next_generation_genomes.append(child)
            attempts += 1
            
        attempts = 0
        while len(next_generation_genomes) < self.pop_size and attempts < 1000:
            immigrant = self.optimizer.generate_random_genome()
            immigrant_tuple = tuple(immigrant)
            if immigrant_tuple not in unique_genomes:
                unique_genomes.add(immigrant_tuple)
                next_generation_genomes.append(immigrant)
            attempts += 1

        # 수렴 방지 및 개체군 유지를 위한 Random Fallback
        while len(next_generation_genomes) < self.pop_size:
            logger.warning("Search space heavily converged. Applying forced random fill to maintain population size.")
            fallback_genome = self.optimizer.generate_random_genome()
            next_generation_genomes.append(fallback_genome)

        return next_generation_genomes