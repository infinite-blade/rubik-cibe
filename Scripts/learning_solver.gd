
class_name LearningSolver
extends RefCounted

# Input size: 48 stickers * 6 one-hot colors = 288 features
const INPUT_SIZE = 288
const HIDDEN_SIZE = 48
const OUTPUT_SIZE = 12
const WEIGHTS_PATH = "user://cube_nn_weights.json"

const ALL_MOVES = [
	[0, 1], [0, -1], [1, 1], [1, -1], [2, 1], [2, -1],
	[3, 1], [3, -1], [4, 1], [4, -1], [5, 1], [5, -1]
]

static var w1: Array = [] # [HIDDEN_SIZE][INPUT_SIZE]
static var b1: Array = [] # [HIDDEN_SIZE]
static var w2: Array = [] # [OUTPUT_SIZE][HIDDEN_SIZE]
static var b2: Array = [] # [OUTPUT_SIZE]
static var is_trained: bool = false

static func init_weights():
	w1.clear(); b1.clear(); w2.clear(); b2.clear()
	for h in HIDDEN_SIZE:
		var row = []
		for i in INPUT_SIZE:
			row.append(randf_range(-0.1, 0.1))
		w1.append(row)
		b1.append(0.0)
	for o in OUTPUT_SIZE:
		var row = []
		for h in HIDDEN_SIZE:
			row.append(randf_range(-0.1, 0.1))
		w2.append(row)
		b2.append(0.0)

# Convert CubeState (48 stickers, each 0..5) into 288 one-hot floats
static func encode_state(s: CubeState) -> PackedFloat32Array:
	var x = PackedFloat32Array()
	x.resize(INPUT_SIZE)
	x.fill(0.0)
	for i in 24:
		x[i * 6 + s.e[i]] = 1.0
	for i in 24:
		x[(24 + i) * 6 + s.c[i]] = 1.0
	return x

static func _forward(x: PackedFloat32Array) -> Dictionary:
	var hidden = PackedFloat32Array()
	hidden.resize(HIDDEN_SIZE)
	for h in HIDDEN_SIZE:
		var sum = b1[h]
		var w_row = w1[h]
		for i in INPUT_SIZE:
			if x[i] > 0.5: # Sparse speedup since x is one-hot!
				sum += w_row[i]
		hidden[h] = max(0.0, sum) # ReLU activation

	var logits = PackedFloat32Array()
	logits.resize(OUTPUT_SIZE)
	var max_l = -1e9
	for o in OUTPUT_SIZE:
		var sum = b2[o]
		var w_row = w2[o]
		for h in HIDDEN_SIZE:
			sum += w_row[h] * hidden[h]
		logits[o] = sum
		if sum > max_l: max_l = sum

	# Softmax probabilities
	var probs = PackedFloat32Array()
	probs.resize(OUTPUT_SIZE)
	var exp_sum = 0.0
	for o in OUTPUT_SIZE:
		probs[o] = exp(logits[o] - max_l)
		exp_sum += probs[o]
	for o in OUTPUT_SIZE:
		probs[o] /= exp_sum

	return { "hidden": hidden, "probs": probs }

# Self-supervised training using reverse scrambles
static func train(samples: int = 800, max_depth: int = 6, lr: float = 0.02) -> Dictionary:
	init_weights()
	var total_loss = 0.0
	var correct = 0

	for step in samples:
		var s = CubeState.new()
		var depth = randi_range(1, max_depth)
		var last_move_idx = -1
		for d in depth:
			var m_idx = randi() % 12
			# Avoid immediate inverse move during scramble generation
			while last_move_idx != -1 and (m_idx ^ 1) == last_move_idx:
				m_idx = randi() % 12
			var mv = ALL_MOVES[m_idx]
			s = s.apply_move(mv[0], mv[1])
			last_move_idx = m_idx

		# Target is the INVERSE of the last move (XOR 1 flips +1 <-> -1 in ALL_MOVES!)
		var target_idx = last_move_idx ^ 1
		var x = encode_state(s)
		var out = _forward(x)
		var probs = out.probs
		var hidden = out.hidden

		total_loss += -log(max(probs[target_idx], 1e-7))
		var best_pred = 0
		for o in OUTPUT_SIZE:
			if probs[o] > probs[best_pred]: best_pred = o
		if best_pred == target_idx:
			correct += 1

		# Backpropagation (Cross-Entropy + Softmax derivative = probs - one_hot)
		var d_logits = PackedFloat32Array()
		d_logits.resize(OUTPUT_SIZE)
		for o in OUTPUT_SIZE:
			d_logits[o] = probs[o] - (1.0 if o == target_idx else 0.0)

		var d_hidden = PackedFloat32Array()
		d_hidden.resize(HIDDEN_SIZE)
		d_hidden.fill(0.0)

		for o in OUTPUT_SIZE:
			var dl = d_logits[o]
			b2[o] -= lr * dl
			for h in HIDDEN_SIZE:
				d_hidden[h] += dl * w2[o][h]
				w2[o][h] -= lr * dl * hidden[h]

		for h in HIDDEN_SIZE:
			if hidden[h] > 0.0: # ReLU derivative
				var dh = d_hidden[h]
				b1[h] -= lr * dh
				for i in INPUT_SIZE:
					if x[i] > 0.5:
						w1[h][i] -= lr * dh

	is_trained = true
	save_weights()
	return {
		"avg_loss": total_loss / samples,
		"accuracy": float(correct) / samples * 100.0,
		"samples": samples
	}

static func save_weights():
	var file = FileAccess.open(WEIGHTS_PATH, FileAccess.WRITE)
	if file:
		file.store_string(JSON.stringify({ "w1": w1, "b1": b1, "w2": w2, "b2": b2 }))

static func load_weights() -> bool:
	if not FileAccess.file_exists(WEIGHTS_PATH):
		return false
	var data = JSON.parse_string(FileAccess.get_file_as_string(WEIGHTS_PATH))
	if data is Dictionary and data.has("w1"):
		w1 = data.w1; b1 = data.b1; w2 = data.w2; b2 = data.b2
		is_trained = true
		return true
	return false

# Solve using NN-Guided Beam Search
static func solve_with_nn(start_state: CubeState, beam_width: int = 25, max_steps: int = 12) -> Dictionary:
	if not is_trained and not load_weights():
		train() # Auto-train once if no saved weights exist

	if start_state.is_solved():
		return { "solved": true, "moves": [], "nodes_expanded": 0 }

	var beam = [{ "state": start_state, "path": [], "score": float(start_state.solved_sticker_score()) }]
	var visited = { start_state.state_key(): true }
	var nodes = 0

	for step in max_steps:
		var candidates = []
		for item in beam:
			var probs = _forward(encode_state(item.state)).probs
			for m_idx in 12:
				var mv = ALL_MOVES[m_idx]
				var next_s = item.state.apply_move(mv[0], mv[1])
				nodes += 1
				var new_path = item.path.duplicate()
				new_path.append(mv)
				if next_s.is_solved():
					return { "solved": true, "moves": new_path, "nodes_expanded": nodes }
				var key = next_s.state_key()
				if not visited.has(key):
					visited[key] = true
					# Combine NN move probability + sticker progress
					var combined_score = probs[m_idx] * 20.0 + next_s.solved_sticker_score()
					candidates.append({ "state": next_s, "path": new_path, "score": combined_score })

		if candidates.is_empty():
			break
		candidates.sort_custom(func(a, b): return a.score > b.score)
		beam = candidates.slice(0, min(beam_width, candidates.size()))

	return { "solved": false, "moves": beam[0].path, "nodes_expanded": nodes }
