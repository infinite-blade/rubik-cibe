class_name SearchSolver
extends RefCounted

const ALL_MOVES = [
	[0, 1], [0, -1], [1, 1], [1, -1], [2, 1], [2, -1],
	[3, 1], [3, -1], [4, 1], [4, -1], [5, 1], [5, -1]
]
const OPPOSITE_FACE = { 0: 4, 4: 0, 1: 3, 3: 1, 2: 5, 5: 2 }
const FACE_MAPS = [
	[0,1,2,3,4,5], [0,2,3,5,4,1], [0,3,5,1,4,2], [0,5,1,2,4,3]
]

static var nodes_expanded: int = 0
static var max_ida_nodes: int = 25000

static func solve_ida_star(start_state: CubeState) -> Dictionary:
	nodes_expanded = 0
	if start_state.is_solved():
		return { "solved": true, "moves": [], "nodes_expanded": 0, "depth": 0 }

	# 1. Try Pure IDA* first (optimal shortest path for shallow scrambles)
	var bound = start_state.heuristic_admissible()
	var path: Array = []
	while nodes_expanded < max_ida_nodes:
		var t = _dfs(start_state, 0, bound, -1, path)
		if t == -1:
			return { "solved": true, "moves": path, "nodes_expanded": nodes_expanded, "depth": path.size() }
		if t == INF:
			break
		bound = t

	# 2. For deep scrambles (10-15+ moves), run Subgoal A* & Macro-BFS Search
	return solve_subgoals(start_state, nodes_expanded)

static func _dfs(state: CubeState, g: int, bound: int, last_face: int, path: Array):
	nodes_expanded += 1
	if nodes_expanded >= max_ida_nodes:
		return INF
	var f = g + state.heuristic_admissible()
	if f > bound:
		return f
	if state.is_solved():
		return -1

	var min_exceeded = INF
	for mv in ALL_MOVES:
		var face = mv[0]
		if path.size() > 0:
			var prev = path[-1]
			if prev[0] == face and prev[1] != mv[1]: continue
			if path.size() >= 2 and path[-2][0] == face and prev[0] == face: continue
			if OPPOSITE_FACE[face] == last_face and face < last_face: continue
		path.append(mv)
		var res = _dfs(state.apply_move(face, mv[1]), g + 1, bound, face, path)
		if res == -1: return -1
		if res < min_exceeded: min_exceeded = res
		path.pop_back()
	return min_exceeded

# ==============================================================================
# Subgoal Graph Search (A* for Cross + Macro-BFS for F2L & Last Layer)
# ==============================================================================
static func solve_subgoals(start_state: CubeState, prev_nodes: int = 0) -> Dictionary:
	var s = start_state.duplicate_state()
	var full_path: Array = []
	var total_nodes = prev_nodes

	# Stage 1: White Cross (edges 8, 9, 10, 11) via A* Search
	var locked_edges: Array = []
	for e_idx in [8, 9, 10, 11]:
		var res = _astar_cross_edge(s, e_idx, locked_edges)
		total_nodes += res.nodes
		s = res.state
		full_path.append_array(res.moves)
		locked_edges.append(e_idx)

	# Stage 2: White Corners (4, 5, 6, 7) via Macro-BFS
	var corner_macros = _build_corner_macros()
	var locked_corners: Array = []
	for c_idx in [4, 5, 6, 7]:
		var res = _bfs_macro(s, corner_macros, func(st):
			return _all_edges_ok(st, locked_edges) and _all_corners_ok(st, locked_corners) and _is_corner_ok(st, c_idx)
		)
		total_nodes += res.nodes
		s = res.state
		full_path.append_array(res.moves)
		locked_corners.append(c_idx)

	# Stage 3: Middle Layer Edges (4, 5, 6, 7) via Macro-BFS
	var mid_macros = _build_mid_macros()
	for e_idx in [4, 5, 6, 7]:
		var res = _bfs_macro(s, mid_macros, func(st):
			return _all_edges_ok(st, locked_edges) and _all_corners_ok(st, locked_corners) and _is_edge_ok(st, e_idx)
		)
		total_nodes += res.nodes
		s = res.state
		full_path.append_array(res.moves)
		locked_edges.append(e_idx)

	# Stage 4: Top Yellow Cross (edges 0..3 have yellow=0 in slot 1)
	var top_cross_macros = [[[0,1]], [[0,-1]], [[2,1],[3,1],[0,1],[3,-1],[0,-1],[2,-1]]]
	var r4 = _bfs_macro(s, top_cross_macros, func(st):
		return _all_edges_ok(st, locked_edges) and _all_corners_ok(st, locked_corners) and _is_top_cross_ok(st)
	)
	total_nodes += r4.nodes; s = r4.state; full_path.append_array(r4.moves)

	# Stage 5: Align Top Edges (0, 1, 2, 3) using Sune macro
	var top_edge_macros = [[[0,1]], [[0,-1]], [[3,1],[0,1],[3,-1],[0,1],[3,1],[0,1],[0,1],[3,-1]]]
	var r5 = _bfs_macro(s, top_edge_macros, func(st):
		return _all_edges_ok(st, [0,1,2,3,4,5,6,7,8,9,10,11]) and _all_corners_ok(st, locked_corners)
	)
	total_nodes += r5.nodes; s = r5.state; full_path.append_array(r5.moves)

	# Stage 6: Position Top Corners (0, 1, 2, 3) using Niklas macro on all 4 sides
	var pos_macros = _build_niklas_macros()
	var r6 = _bfs_macro(s, pos_macros, func(st):
		return _all_edges_ok(st, [0,1,2,3,4,5,6,7,8,9,10,11]) and _all_corners_ok(st, locked_corners) and _top_corners_positioned(st)
	)
	total_nodes += r6.nodes; s = r6.state; full_path.append_array(r6.moves)

	# Stage 7: Orient Top Corners using pure 2-corner twist commutators
	var twist_macros = _build_twist_macros()
	var r7 = _bfs_macro(s, twist_macros, func(st): return st.is_solved())
	total_nodes += r7.nodes; s = r7.state; full_path.append_array(r7.moves)

	return {
		"solved": s.is_solved(),
		"moves": full_path,
		"nodes_expanded": total_nodes,
		"depth": full_path.size()
	}

# --- A* Search for one Cross Edge ---
static func _astar_cross_edge(start: CubeState, target: int, locked: Array) -> Dictionary:
	if _is_edge_ok(start, target):
		return { "state": start, "moves": [], "nodes": 0 }
	var open = [{ "state": start, "moves": [], "g": 0, "f": _cross_h(start, target, locked) }]
	var visited = { start.state_key(): 0 }
	var nodes = 0
	while open.size() > 0 and nodes < 20000:
		var best_i = 0
		for i in range(1, open.size()):
			if open[i].f < open[best_i].f: best_i = i
		var cur = open[best_i]
		open.remove_at(best_i)
		nodes += 1
		if _is_edge_ok(cur.state, target) and _all_edges_ok(cur.state, locked):
			return { "state": cur.state, "moves": cur.moves, "nodes": nodes }
		for mv in ALL_MOVES:
			var ns = cur.state.apply_move(mv[0], mv[1])
			var ng = cur.g + 1
			var k = ns.state_key()
			if not visited.has(k) or ng < visited[k]:
				visited[k] = ng
				var nm = cur.moves.duplicate()
				nm.append(mv)
				open.append({ "state": ns, "moves": nm, "g": ng, "f": ng + _cross_h(ns, target, locked) })
	return { "state": start, "moves": [], "nodes": nodes }

static func _cross_h(s: CubeState, target: int, locked: Array) -> int:
	var broken = 0
	for idx in locked:
		if not _is_edge_ok(s, idx): broken += 1
	var dist = 0
	if not _is_edge_ok(s, target):
		var c0 = CubeState.EDGE_MAP[target][0]
		var c1 = CubeState.EDGE_MAP[target][1]
		var pos = 0
		for i in 12:
			var a = s.e[i * 2]; var b = s.e[i * 2 + 1]
			if (a == c0 and b == c1) or (a == c1 and b == c0):
				pos = i; break
		if pos <= 3: dist = 1
		elif pos <= 7: dist = 2
		elif pos == target: dist = 3
		else: dist = 2
	return broken * 3 + dist

# --- BFS Search over Macro-Operators ---
static func _bfs_macro(start: CubeState, macros: Array, goal_fn: Callable) -> Dictionary:
	if goal_fn.call(start):
		return { "state": start, "moves": [], "nodes": 0 }
	var queue = [{ "state": start, "moves": [] }]
	var visited = { start.state_key(): true }
	var nodes = 0
	while queue.size() > 0 and nodes < 6000:
		var cur = queue.pop_front()
		for mac in macros:
			var ns = cur.state
			for mv in mac:
				ns = ns.apply_move(mv[0], mv[1])
			nodes += 1
			var nm = cur.moves.duplicate()
			nm.append_array(mac)
			if goal_fn.call(ns):
				return { "state": ns, "moves": nm, "nodes": nodes }
			var k = ns.state_key()
			if not visited.has(k):
				visited[k] = true
				queue.append({ "state": ns, "moves": nm })
	return { "state": start, "moves": [], "nodes": nodes }

# --- Macro Generators (rotated around Y axis using FACE_MAPS) ---
static func _rot_seq(seq: Array, side: int) -> Array:
	var out = []
	for mv in seq:
		out.append([FACE_MAPS[side][mv[0]], mv[1]])
	return out

static func _build_corner_macros() -> Array:
	var out: Array = [[[0, 1]], [[0, -1]]]
	var t_left = [[1,-1],[0,-1],[1,1]]
	var t_front = [[2,1],[0,1],[2,-1]]
	for side in 4:
		out.append(_rot_seq(t_left, side))
		out.append(_rot_seq(t_front, side))
	return out

static func _build_mid_macros() -> Array:
	var out: Array = [[[0, 1]], [[0, -1]]]
	var m_right = [[0,1],[3,1],[0,-1],[3,-1],[0,-1],[2,-1],[0,1],[2,1]]
	var m_left = [[0,-1],[1,-1],[0,1],[1,1],[0,1],[2,1],[0,-1],[2,-1]]
	for side in 4:
		out.append(_rot_seq(m_right, side))
		out.append(_rot_seq(m_left, side))
	return out

static func _build_niklas_macros() -> Array:
	var out: Array = [[[0, 1]], [[0, -1]]]
	var niklas = [[0,1],[3,1],[0,-1],[1,-1],[0,1],[3,-1],[0,-1],[1,1]]
	for side in 4:
		out.append(_rot_seq(niklas, side))
	return out

static func _build_twist_macros() -> Array:
	var out: Array = []
	var spin = [[3,-1],[4,-1],[3,1],[4,1],[3,-1],[4,-1],[3,1],[4,1]]
	var spin_inv = [[4,-1],[3,-1],[4,1],[3,1],[4,-1],[3,-1],[4,1],[3,1]]
	var comm1 = spin + [[0,1]] + spin_inv + [[0,-1]]
	var comm2 = spin + [[0,-1]] + spin_inv + [[0,1]]
	var comm3 = spin + [[0,1],[0,1]] + spin_inv + [[0,1],[0,1]]
	for side in 4:
		out.append(_rot_seq(comm1, side))
		out.append(_rot_seq(comm2, side))
		out.append(_rot_seq(comm3, side))
	return out

# --- State Check Helpers ---
static func _is_edge_ok(s: CubeState, idx: int) -> bool:
	return s.e[idx * 2] == CubeState.EDGE_MAP[idx][0] and s.e[idx * 2 + 1] == CubeState.EDGE_MAP[idx][1]

static func _is_corner_ok(s: CubeState, idx: int) -> bool:
	return s.c[idx * 3] == CubeState.CORNER_MAP[idx][0] and s.c[idx * 3 + 1] == CubeState.CORNER_MAP[idx][1] and s.c[idx * 3 + 2] == CubeState.CORNER_MAP[idx][2]

static func _all_edges_ok(s: CubeState, list: Array) -> bool:
	for idx in list:
		if not _is_edge_ok(s, idx): return false
	return true

static func _all_corners_ok(s: CubeState, list: Array) -> bool:
	for idx in list:
		if not _is_corner_ok(s, idx): return false
	return true

static func _is_top_cross_ok(s: CubeState) -> bool:
	for idx in [0, 1, 2, 3]:
		if s.e[idx * 2 + 1] != 0: return false
	return true

static func _top_corners_positioned(s: CubeState) -> bool:
	for idx in [0, 1, 2, 3]:
		var t = CubeState.CORNER_MAP[idx]
		var cur = [s.c[idx * 3], s.c[idx * 3 + 1], s.c[idx * 3 + 2]]
		if not (cur.has(t[0]) and cur.has(t[1]) and cur.has(t[2])): return false
	return true
