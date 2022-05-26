import tensorflow as tf
import numpy as np
import time

all_shortest_paths = None # global variable
all_simple_paths = None
all_simple_masks = None

def simple_paths_between(source, target, history):
    if source in history:
        return []
    
    if source == target:
        return [history + [target]]

    source_row = source // 4
    target_row = target // 4
    source_col = source % 4
    target_col = target % 4
    candidate_moves = []
    if 3 > source_row:
        candidate_moves.append(source + 4)
    if 0 < source_row:
        candidate_moves.append(source - 4)
    if 3 > source_col:
        candidate_moves.append(source + 1)
    if 0 < source_col:
        candidate_moves.append(source - 1)

    result = []
    history2 = history + [source]
    for c in candidate_moves:
        if c not in history:
            paths = simple_paths_between(c, target, history2)
            result += paths
    return result



def shortest_paths_between(source, target):
    if source == target:
        return [[target]]

    source_row = source // 4
    target_row = target // 4
    source_col = source % 4
    target_col = target % 4
    candidate_moves = []
    if target_row > source_row:
        candidate_moves.append(source + 4)
    elif target_row < source_row:
        candidate_moves.append(source - 4)
    if target_col > source_col:
        candidate_moves.append(source + 1)
    elif target_col < source_col:
        candidate_moves.append(source - 1)

    result = []
    for c in candidate_moves:        
        tmp = shortest_paths_between(c, target)
        paths = [[source] + p for p in tmp]
        result += paths
    return result

def shortest_paths():
    result = []
    for source in range(16):
        for target in range(source+1, 16):
            paths = shortest_paths_between(source, target)
            result += paths
    return result

def path_to_edges(path):
    edgemap = {
        (0,1): 0,
        (0,4): 12,
        (1,2): 1,
        (1,5): 15,
        (2,3): 2,
        (2,6): 18,
        (3,7): 21,
        (4,5): 3,
        (4,8): 13,
        (5,6): 4,
        (5,9): 16,
        (6,7): 5,
        (6,10): 19,
        (7,11): 22,
        (8,9): 6,
        (8,12): 14,
        (9,10): 7,
        (9,13): 17,
        (10,11): 8,
        (10,14): 20,
        (11,15): 23,
        (12,13): 9,
        (13,14): 10,
        (14,15): 11
    }

    edges = []
    for i in range(len(path)-1):
        if path[i] < path[i+1]:
            s = path[i]
            t = path[i+1]
        else:
            s = path[i+1]
            t = path[i]
        edges.append(edgemap[(s,t)])
    return edges
    

    
# 4x4 grid
# return a list of all paths
# return (path_cnt, 24)
def get_shortest_paths():
    global all_shortest_paths
    if all_shortest_paths is None:
        paths = shortest_paths()
        l = len(paths)
        edge_matrix = np.zeros((l,24))
        for i, p in enumerate(paths):
            edges = path_to_edges(p)
            edge_matrix[i][edges] = 1
        all_shortest_paths = tf.constant(edge_matrix, dtype=tf.float32)
        
    return all_shortest_paths


# 4x4 grid
# return a list of all paths along with source-target mask
# return (path_cnt, 24), (path_cnt, 16)
def get_simple_paths():
    global all_simple_paths, all_simple_mask
    if all_simple_paths is None or all_simple_mask is None:
        mask_list = []
        edge_list = []
        for source in range(16):
            for target in range(source+1, 16):
                paths = simple_paths_between(source, target,[])
                l = len(paths)
                edge_matrix = np.zeros((l, 24))
                masks = np.zeros((l, 16))
                masks[:,source] = 1
                masks[:,target] = 1
                for i, p in enumerate(paths):
                    edges = path_to_edges(p)
                    edge_matrix[i][edges] = 1
                mask_list.append(masks)
                edge_list.append(edge_matrix)
        all_simple_paths = np.concatenate(edge_list, axis=0)
        all_simple_mask = np.concatenate(mask_list, axis=0)
        all_simple_paths = tf.constant(all_simple_paths, dtype=tf.float32)
        all_simple_mask = tf.constant(all_simple_mask, dtype=tf.float32)
    return all_simple_paths, all_simple_mask

def get_relevant_paths(endpoints, removed, filter_removed):
    paths, mask = get_simple_paths()
    paths = tf.expand_dims(paths, 0) # 1 * paths * 24

    mask = tf.expand_dims(mask, 0) # 1 * paths * 16
    endpoints = tf.expand_dims(endpoints, axis=1), # bs * 1 * 16
    endpoints = endpoints[0] # TODO why
    mask = tf.cast(tf.math.equal(tf.reduce_sum(mask * endpoints, axis=2), 2), dtype=tf.float32) # bs * paths

    if filter_removed:
        removed = tf.expand_dims(removed, 1) # bs * 1 * 24
        paths_removed = tf.reduce_max(paths * removed, axis=2) # bs * paths
        paths_allowed = 1-paths_removed
        mask = mask * paths_allowed

    return paths, mask

# logits: bs * 24
# endpoints: bs * 16
# removed: bs * 24 binary
def prp_loss(logits, endpoints, removed):
    paths, mask = get_relevant_paths(endpoints, removed, filter_removed=False) # (1 * paths * 24), (bs * paths)

    probs = tf.nn.sigmoid(logits)
    probs = tf.expand_dims(probs, 1) #bs * 1 * 24
    paths_prob = tf.reduce_prod(paths * probs + (1-paths) * (1-probs), axis=2) # bs * paths

    n = 1.0 - tf.reduce_sum(paths_prob * mask)
    log_n = tf.log(tf.maximum(1e-5, n))

    k = tf.reduce_sum(mask, axis=1, keepdims=True)
    log_d = 1 / k * tf.reduce_sum(mask * tf.log(tf.maximum(1e-10, paths_prob)), axis=1)
    loss = log_n - log_d
    return loss

