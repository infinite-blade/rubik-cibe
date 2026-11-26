class_name CubeState
extends RefCounted

# Flattened sticker arrays:
# e has 24 ints (12 edges * 2 stickers each)
# c has 24 ints (8 corners * 3 stickers each)
var e: PackedInt32Array
var c: PackedInt32Array

# Static permutation tables shared across all CubeState instances
# edge_perm[face][k] tells us which old sticker moves into slot k on a +1 turn
static var edge_perm: Array = []
static var edge_perm_inv: Array = []
static var corner_perm: Array = []
static var corner_perm_inv: Array = []
static var tables_ready: bool = false

func _init():
	e.resize(24)
	c.resize(24)
	reset_solved()

func reset_solved():
	for i in 12:
		e[i * 2] = ColorMap_EDGE(i, 0)
		e[i * 2 + 1] = ColorMap_EDGE(i, 1)
	for i in 8:
		c[i * 3] = ColorMap_CORNER(i, 0)
		c[i * 3 + 1] = ColorMap_CORNER(i, 1)
		c[i * 3 + 2] = ColorMap_CORNER(i, 2)

# Helper constants matching ColorMap.EDGE_FACE_MAP and CORNER_FACE_MAP
const EDGE_MAP = [[5,0],[1,0],[3,0],[2,0],[5,1],[1,2],[3,2],[3,5],[1,4],[2,4],[3,4],[5,4]]
const CORNER_MAP = [[0,1,5],[0,5,3],[0,2,1],[0,3,2],[5,1,4],[1,2,4],[2,3,4],[3,5,4]]

static func ColorMap_EDGE(i: int, j: int) -> int:
	return EDGE_MAP[i][j]

static func ColorMap_CORNER(i: int, j: int) -> int:
	return CORNER_MAP[i][j]

# Run ONCE at startup using BigCube to automatically extract exact 3D move rules
static func build_tables_from_big_cube(bc: BigCube):
	if tables_ready:
		return
	edge_perm.resize(6)
	edge_perm_inv.resize(6)
	corner_perm.resize(6)
	corner_perm_inv.resize(6)

	for face in 6:
		bc.reset()
		bc.rotate_face_immediate(face, 1)
		var ec = bc.get_edge_colors()
		var cc = bc.get_corner_colors()

		var ep = PackedInt32Array()
		var ep_inv = PackedInt32Array()
		ep.resize(24)
		ep_inv.resize(24)

		for new_i in 12:
			var c0 = ec[new_i][EDGE_MAP[new_i][0]]
			var c1 = ec[new_i][EDGE_MAP[new_i][1]]
			for old_i in 12:
				if (EDGE_MAP[old_i][0] == c0 and EDGE_MAP[old_i][1] == c1) or \
				   (EDGE_MAP[old_i][0] == c1 and EDGE_MAP[old_i][1] == c0):
					var old_slot0 = old_i * 2 + (0 if EDGE_MAP[old_i][0] == c0 else 1)
					var old_slot1 = old_i * 2 + (1 if EDGE_MAP[old_i][0] == c0 else 0)
					ep[new_i * 2] = old_slot0
					ep[new_i * 2 + 1] = old_slot1
					ep_inv[old_slot0] = new_i * 2
					ep_inv[old_slot1] = new_i * 2 + 1
					break

		var cp = PackedInt32Array()
		var cp_inv = PackedInt32Array()
		cp.resize(24)
		cp_inv.resize(24)

		for new_i in 8:
			var cols = [
				cc[new_i][CORNER_MAP[new_i][0]],
				cc[new_i][CORNER_MAP[new_i][1]],
				cc[new_i][CORNER_MAP[new_i][2]]
			]
			for old_i in 8:
				var target = CORNER_MAP[old_i]
				if cols.has(target[0]) and cols.has(target[1]) and cols.has(target[2]):
					for j in 3:
						var old_j = target.find(cols[j])
						var new_slot = new_i * 3 + j
						var old_slot = old_i * 3 + old_j
						cp[new_slot] = old_slot
						cp_inv[old_slot] = new_slot
					break

		edge_perm[face] = ep
		edge_perm_inv[face] = ep_inv
		corner_perm[face] = cp
		corner_perm_inv[face] = cp_inv

	bc.reset()
	tables_ready = true

# Copy the current 2D ColorMap state into a CubeState for the AI to solve
static func from_color_map(cmap) -> CubeState:
	var s = CubeState.new()
	for i in 12:
		s.e[i * 2] = cmap.edges[i][0]
		s.e[i * 2 + 1] = cmap.edges[i][1]
	for i in 8:
		s.c[i * 3] = cmap.corners[i][0]
		s.c[i * 3 + 1] = cmap.corners[i][1]
		s.c[i * 3 + 2] = cmap.corners[i][2]
	return s

func duplicate_state() -> CubeState:
	var s = CubeState.new()
	s.e = e.duplicate()
	s.c = c.duplicate()
	return s

func apply_move(face: int, dir: int) -> CubeState:
	var next = CubeState.new()
	var ep = edge_perm[face] if dir == 1 else edge_perm_inv[face]
	var cp = corner_perm[face] if dir == 1 else corner_perm_inv[face]
	for k in 24:
		next.e[k] = e[ep[k]]
		next.c[k] = c[cp[k]]
	return next

func is_solved() -> bool:
	for i in 12:
		if e[i * 2] != EDGE_MAP[i][0] or e[i * 2 + 1] != EDGE_MAP[i][1]:
			return false
	for i in 8:
		if c[i * 3] != CORNER_MAP[i][0] or c[i * 3 + 1] != CORNER_MAP[i][1] or c[i * 3 + 2] != CORNER_MAP[i][2]:
			return false
	return true

func state_key() -> String:
	return str(e) + "|" + str(c)

# Admissible heuristic for IDA* (lower bound on moves remaining)
func heuristic_admissible() -> int:
	var wrong = 0
	for i in 12:
		if e[i * 2] != EDGE_MAP[i][0]: wrong += 1
		if e[i * 2 + 1] != EDGE_MAP[i][1]: wrong += 1
	for i in 8:
		if c[i * 3] != CORNER_MAP[i][0]: wrong += 1
		if c[i * 3 + 1] != CORNER_MAP[i][1]: wrong += 1
		if c[i * 3 + 2] != CORNER_MAP[i][2]: wrong += 1
	# One face turn moves at most 20 edge/corner stickers (8 edge + 12 corner)
	return int(ceil(wrong / 20.0))

# Weighted score for GA / Beam Search (counts solved stickers out of 48)
func solved_sticker_score() -> int:
	var score = 0
	for i in 12:
		if e[i * 2] == EDGE_MAP[i][0]: score += 1
		if e[i * 2 + 1] == EDGE_MAP[i][1]: score += 1
	for i in 8:
		if c[i * 3] == CORNER_MAP[i][0]: score += 1
		if c[i * 3 + 1] == CORNER_MAP[i][1]: score += 1
		if c[i * 3 + 2] == CORNER_MAP[i][2]: score += 1
	return score
