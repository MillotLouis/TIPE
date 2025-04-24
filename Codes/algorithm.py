from heapq import *
from network import *

def distances(net:NetworkSimulator, source, p, a, b):
    """Algorithme de routage prenant en compte la batterie restante des noeuds
    p: seuil de batterie à partir duquel on évite de prendre ce noeud
    a et b coeffs pour calculer poids : a*distance + b*1/batterie
    renvoie une liste de tuples (distance, predecesseur) pour chaque noeud
    """
    infos = {noeud: (float('inf'), None) for noeud in net.get_neighbors(source)}
    infos[source] = (0, source)
    pqueue = [(0, source)]

    while len(pqueue)>0:
        current_distance, current_node = heappop(pqueue)

        if current_distance > infos[current_node][0]:
            continue

        for neighbor in net.get_neighbors(current_node):
            battery = net.get_battery(neighbor)
            if battery < p:
                weight = 10**5 #Distances en mètre donc 100km est largement supérieur à distance max
            else:
                distance = net.get_distance(current_node, neighbor)
                weight = a * distance + b * (1 / battery)
                new_distance = current_distance + weight

            if new_distance < infos[neighbor][0]:
                infos[neighbor] = (new_distance, current_node)
                heappush(pqueue, (new_distance, neighbor))

    return [(node, (infos[node][0], infos[node][1])) for node in infos]

def updatetable(net:NetworkSimulator, p, a, b):
    for node1 in net.G:
        infos = distances(net, node1, p, a, b)
        for node2, (dist, _) in infos:
            path = []
            current = node2
            while current is not None and current != node1:
                path.insert(0, current)
                current = infos[current][1]
            net.table[node1][node2] = (dist, path)


def get_path_dijkstra(net,source,dest):
    return net.table[source][dest][1]