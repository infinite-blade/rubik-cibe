class_name OptimizationSolver
extends RefCounted

const ALL_MOVES = [
	[0, 1], [0, -1], [1, 1], [1, -1], [2, 1], [2, -1],
	[3, 1], [3, -1], [4, 1], [4, -1], [5, 1], [5, -1]
]

# 1. Genetic Algorithm Solver (searches move sequences via Evolution)
static func solve_ga(start_state: CubeState, pop_size: int = 60, generations: int = 80, seq_len: int = 16) -> Dictionary:
	if start_state.is_solved():
		return { "solved": true, "moves": [], "nodes_expanded": 0, "best_fitness": 48.0 }

	var population = []
	for p in pop_size:
		var chrom = []
		for g in seq_len:
			chrom.append(ALL_MOVES[randi() % 12])
		population.append(chrom)

	var best_moves = []
	var best_fitness = -1.0
	var evals = 0

	for gen in generations:
		var scored = []
		for chrom in population:
			var res = _evaluate(start_state, chrom)
			evals += chrom.size()
			if res.solved:
				return {
					"solved": true,
					"moves": compress_moves(res.used_moves),
					"nodes_expanded": evals,
					"best_fitness": 100.0
				}
			if res.fitness > best_fitness:
				best_fitness = res.fitness
				best_moves = res.used_moves
			scored.append({ "chrom": chrom, "fitness": res.fitness })

		scored.sort_custom(func(a, b): return a.fitness > b.fitness)

		# Elitism: keep top 20%
		var next_pop = []
		var elite_count = max(2, int(pop_size * 0.2))
		for i in elite_count:
			next_pop.append(scored[i].chrom.duplicate(true))

		# Crossover & Mutation for remaining 80%
		while next_pop.size() < pop_size:
			var p1 = scored[randi() % elite_count].chrom
			var p2 = scored[randi() % pop_size].chrom
			var cut = randi_range(1, seq_len - 1)
			var child = p1.slice(0, cut) + p2.slice(cut, seq_len)
			# Mutation (15% chance per gene)
			for i in child.size():
				if randf() < 0.15:
					child[i] = ALL_MOVES[randi() % 12]
			next_pop.append(child)

		population = next_pop

	return {
		"solved": false,
		"moves": compress_moves(best_moves),
		"nodes_expanded": evals,
		"best_fitness": best_fitness
	}

static func _evaluate(start_state: CubeState, chrom: Array) -> Dictionary:
	var s = start_state
	var best_score = s.solved_sticker_score()
	var best_prefix_len = 0
	for i in chrom.size():
		s = s.apply_move(chrom[i][0], chrom[i][1])
		if s.is_solved():
			return { "solved": true, "fitness": 100.0 - i * 0.5, "used_moves": chrom.slice(0, i + 1) }
		var sc = s.solved_sticker_score()
		if sc > best_score:
			best_score = sc
			best_prefix_len = i + 1
	return {
		"solved": false,
		"fitness": float(best_score) - best_prefix_len * 0.05,
		"used_moves": chrom.slice(0, best_prefix_len)
	}

# 2. Algebraic Move Sequence Optimizer (removes redundant turns!)
# Cancels [F, 1] + [F, -1] and simplifies [F, 1] + [F, 1] + [F, 1] -> [F, -1]
# Axis lookup for FACES: 0=UP, 1=LEFT, 2=FRONT, 3=RIGHT, 4=DOWN, 5=BACK
const AXIS_OF = [0, 1, 2, 1, 0, 2] # U/D=0, L/R=1, F/B=2

# Automatically optimizes any move sequence (cancels inverses, triples, & opposite-face commutes)
static func compress_moves(moves: Array) -> Array:
	var current = moves.duplicate(true)
	var changed = true

	while changed:
		changed = false
		var next_moves: Array = []
		var i = 0
		while i < current.size():
			var axis = AXIS_OF[current[i][0]]
			# Group consecutive moves on the same axis (e.g. U and D)
			var counts = {}
			var j = i
			while j < current.size() and AXIS_OF[current[j][0]] == axis:
				var f = current[j][0]
				var d = current[j][1]
				counts[f] = counts.get(f, 0) + d
				j += 1

			# Rebuild minimal moves for this axis group
			var group_out: Array = []
			var faces = counts.keys()
			faces.sort()
			for f in faces:
				var turns = posmod(counts[f], 4)
				if turns == 1:
					group_out.append([f, 1])
				elif turns == 2:
					group_out.append([f, 1])
					group_out.append([f, 1])
				elif turns == 3:
					group_out.append([f, -1])

			if group_out.size() < (j - i):
				changed = true
			next_moves.append_array(group_out)
			i = j

		current = next_moves

	return current
