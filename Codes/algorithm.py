from heapq import *
from network import *

def algo(net: NetworkSimulator, source, p, a, b):
    """Algorithme de routage prenant en compte la batterie restante des noeuds
    p: seuil de batterie à partir duquel on évite de prendre ce noeud
    a et b coeffs pour calculer poids : a*distance + b*1/batterie
    renvoie une liste de tuples (distance, predecesseur) pour chaque noeud
    """
    infos = {noeud: (float('inf'), None, "bleu") for noeud in net.get_neighbors(source)}
    infos[source] = (0, source, "vert")
    verts = [(0, source)]

    while verts:
        # Extract the node with the smallest distance
        current_distance, current_node = heappop(verts)
        infos[current_node] = (current_distance, infos[current_node][1], "red")

        # Process neighbors
        for neighbor in net.get_neighbors(current_node):
            if infos[neighbor][2] == "red":
                continue

            battery = net.get_battery(neighbor)
            if battery < p:
                weight = float('inf')
            else:
                distance = net.get_distance(current_node, neighbor)
                weight = a * distance + b * (1 / battery)
                new_distance = current_distance + weight

            if new_distance < infos[neighbor][0]:
                infos[neighbor] = (new_distance, current_node, "vert")
                heappush(verts, (new_distance, neighbor))

    return [(node, (infos[node][0], infos[node][1])) for node in infos]
