import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.image as mpimg
import random as rd 
import numpy as np 
import json ,os, sys 

class Erg1_TemMl:
    def __init__(self, main_file):
        """
        main_file: the Operating project file
        commands: {
        Graph.load # Loads the data
        Graph.create
        Graph.save
        }"""
        self.main_file = main_file

    def __repr__(self):
        print(self.main_file) 
        
    def load_image_to_matrix(self, image_path):
        image_matrix = mpimg.imread(image_path)
        return image_matrix
    
    # def SaveResults(self,OutFile,Data):
    #     OutFile = 1
    #     return 
    
    def power_method(self, A_mat, n_eigonvalues, e_pres=10**-9 , k_max=10000):
        print("__power method__") 
        if A_mat.shape[0] == A_mat.shape[1] and n_eigonvalues <= A_mat.shape[1] :
                Eigenvalues = []
                N = A_mat.shape[0]
                Out = {}
                for i in range(n_eigonvalues): # Για όσα ιδιοδιανύσματα θέλουμε να βρούμε θα δημιουργήσουμε ενα 
                    x0 = np.random.randint(1,100,A_mat.shape[1]) # τυχαίο "κανονικό" διάνυσμα στον Ρ**ν
                    x0 = x0 / np.linalg.norm(x0) 
                    
                    def itterations(xk): # Θα καλέσουμε αυτην τη συνάρτιση αναδρομικά για να φτάσουμε την δεδομένη ακρίβεια
                        xk_prev = np.copy(xk) 
                        xk = A_mat @ xk_prev
                        xk = xk / np.linalg.norm(xk) 
                        dk = np.linalg.norm(xk - xk_prev) 
                        eigenvalue = (xk.T @ A_mat @ xk) / (xk.T @ xk)
                        return {"xk":xk, "dk":dk, "eigenvalue":eigenvalue} # Τελικά το κάθε επόμενο xk+1 "σπρόχνει" την πρόβλεψη του ιδιοδιαν.
                                                                            # να "στριμωχτεί" σε ένα περιθόριο dk!
                    # Οι υπερπαράμετροι του μοντέλου μας
                    k, dk = 0,1 
                    init_set = itterations(x0)
                    eigenvalue = init_set["eigenvalue"]
                    while dk > e_pres and k < k_max : 
                        k += 1 
                        xk = init_set["xk"]
                        dk = init_set["dk"] 
                        init_set = itterations(xk)
                        eigenvalue = init_set["eigenvalue"]
                    Eigenvalues.append(eigenvalue)
                    eigenvector = np.linalg.solve(A_mat - eigenvalue * np.eye(N), np.zeros(N)) # (Α-λΙ)χ = 0 
                    Out[eigenvalue] = eigenvector 
                    # Deflation step : χρειαζόμαστε μια μέθοδο να παίρνουμε τον επόμενης μικρότερης τάξη πίνακα Α.
                A_mat = A_mat - Eigenvalues[-1] * np.outer(init_set["xk"], init_set["xk"])
                # print(A_mat)  
                return Out 
        else:
            print("This is not an orthogonal matrix or you ask for too many eigenvalues.")
            return 
    

    # Graph section ____________________________:
    # def load_data(self, filepath):
    #     data = json.load(filepath) 
    #     return data 

    def make_graph(self, graph_data):
        G = nx.DiGraph()

        # Step 3: Add nodes and edges to the graph
        for node, neighbors in graph_data.items():
            G.add_node(node)
            G.add_edges_from((node, neighbor) for neighbor in neighbors)

        # Customize the appearance
        pos = nx.spring_layout(G, seed=42, k=0.3)  # Adjust k for more spacing
        edge_labels = {(node, neighbor): f"{node}->{neighbor}" for node, neighbors in graph_data.items() for neighbor in neighbors}

        # Define node colors based on node degree
        node_colors = [G.degree(node) for node in G.nodes()]

        # Draw nodes and edges
        nx.draw_networkx_nodes(G, pos, node_size=700, node_color=node_colors, cmap=plt.cm.Blues, alpha=0.7)
        nx.draw_networkx_edges(G, pos, width=2, edge_color="gray", alpha=0.7)
        nx.draw_networkx_labels(G, pos, font_size=10, font_color="black")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")

        # Set plot limits and display
        plt.title("Directed Graph Visualization")
        plt.axis("off")
        plt.colorbar(nx.draw_networkx_nodes(G, pos, node_size=1, node_color=node_colors, cmap=plt.cm.Blues))
        plt.show()

    def createMarkov(self, graph_data): # Aπλή python εδώ δεν έχω πολλά να σχολιάσω. Ο αλγόριθμος είναι απλός σχετικά
        """ Εισάγουμε τα δεδομένα σε μορφή json και παίρνουμε πίσω τον M = d*A+((1-d)/N)*B """    
        print("__createMarkov__") 
        def create_A(): # Ανδρομική διαδικασία 
            len_links = {key: len(value) for key, value in graph_data.items()} 
            Columns = []
            Nodes = list(graph_data.keys())
            Edges = list(graph_data.values())
            # print(Nodes, Edges) 
            for node in Nodes: # Για κάθε ιστοσελίδα
                try:
                    const = 1/len_links[node]  # 1/αριθμός των σχέσεων κάθε σελίδας 
                except: # Προσοχή στο 0
                    const = 0 
                temp_column = [] 
                for edge in range(1,len(Edges)+1): # Σχεσεις κάθε ιστοσελίδας
                    if edge in graph_data[node]:
                        temp_column.append(const) 
                    else:
                        temp_column.append(0) 
                Columns.append(temp_column)
            return Columns 
        
        n = len(list(graph_data.keys()))
        d = .85
        print(f"We have a {n}*{n} matrix.")
        B = np.ones(n)
        A = np.array(create_A()).T 
        # print(A,"\n")  
        M = d*A + ((1-d)/n)*B
        print("__Markov created__")
        return M

    
erg_inst = Erg1_TemMl(os.getcwd())
A = np.random.rand(3, 3)
# eigenvalues, eigenvectors = np.linalg.eig(A) 
# eigen_dict = dict.fromkeys(eigenvalues, None)

# # Assign corresponding eigenvectors to each eigenvalue
# for eigenvalue, eigenvector in zip(eigenvalues, eigenvectors.T):
#     eigen_dict[eigenvalue] = eigenvector
# print(eigen_dict) 
# # We see some complex eigenvlaues - eigenvectors
# print(erg_inst.power_method(A, 3)) 

    


