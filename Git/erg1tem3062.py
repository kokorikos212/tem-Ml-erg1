# Tem 3062 Μαυρουδής Παναγιώτης Εργαστήριο 1
# Η παρούσα βιβλιοθήκη απευθήνεται σε καθένα από τα προβλήματα του πρώτου εργαστηρίου του μαθήματος μηχανικής μάθησης. 
# Στο αντίστοιχο .ipynb αρχείο που επισυνάπτεται γίνεται χρήση αυτής της βιβλιοθήκης ως προς την επίλυση των προβλημάτων. 

import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.image as mpimg
import random as rd 
import numpy as np 
import json ,os, sys 
# Those libraries are needed for the ploting class 
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

    def generateT(self, n):
        if n <= 0:
            raise ValueError("Input must be a positive integer.")

        matrix = np.zeros((n, n))

        np.fill_diagonal(matrix, 2)

        np.fill_diagonal(matrix[1:], -1)
        np.fill_diagonal(matrix[:, 1:], -1)

        return matrix

    def power_method(self, A_mat, x0=None, n_eigonvalues=1, e_pres=10**-9 , k_max=10000):
        """Power method 
        A_mat = n*n matrix 
        n_eigonvalues = The number of eigenvalues you want to get.
        e_pres = pressision 
        k_max = max itterations 
        """
        print("__power method__") 
        if A_mat.shape[0] == A_mat.shape[1] and n_eigonvalues <= A_mat.shape[1] :
                Eigenvalues = []
                N = A_mat.shape[0]
                Out = {}
                try:
                    if not x0.any():
                        x0 = np.random.randint(1,100,A_mat.shape[1]) # τυχαίο "κανονικό" διάνυσμα στον Ρ**ν
                        x0 = x0 / np.linalg.norm(x0)   
                except:
                    x0 = np.random.randint(1,100,A_mat.shape[1]) # τυχαίο "κανονικό" διάνυσμα στον Ρ**ν
                    x0 = x0 / np.linalg.norm(x0) 

                for i in range(n_eigonvalues): # Για όσα ιδιοδιανύσματα θέλουμε να βρούμε 
                    def itterations(xk): # Θα καλέσουμε αυτην τη συνάρτιση αναδρομικά για να φτάσουμε την δεδομένη ακρίβεια
                        xk_prev = np.copy(xk) 
                        xk = A_mat @ xk_prev
                        xk = xk / np.linalg.norm(xk) 
                        dk = np.linalg.norm(xk - xk_prev) 
                        return {"eigenvector":xk, "dk":dk} # Τελικά το κάθε επόμενο xk+1 "σπρόχνει" την πρόβλεψη του ιδιοδιαν.
                                                                            # να "στριμωχτεί" σε ένα περιθόριο dk!
                    # Οι υπερπαράμετροι του μοντέλου μας
                    k, dk = 0,1 
                    Dk = []
                    init_set = itterations(x0)
                    while dk > e_pres and k < k_max : 
                        k += 1 
                        xk = init_set["eigenvector"]
                        xk = xk/np.linalg.norm(xk) 
                        dk = init_set["dk"] 
                        init_set = itterations(xk)
                        eigenvector = init_set["eigenvector"]
                    Dk.append(dk) 
                    eigenvalue = (xk.T @ A_mat @ xk) / (xk.T @ xk) 
                    Eigenvalues.append(abs(eigenvalue)) 
                    norm_eigenvector = eigenvector / np.linalg.norm(eigenvector)
                    Out[eigenvalue] = norm_eigenvector 
                    # print(eigenvector) 
                    # Deflation step : χρειαζόμαστε μια μέθοδο να παίρνουμε τον επόμενης μικρότερης τάξη πίνακα Α.
                    A_mat = A_mat - Eigenvalues[-1] * np.outer(init_set["eigenvector"], init_set["eigenvector"])
                    A_mat = A_mat/np.linalg.norm(A_mat)  
                    # print(A_mat)  
                    # print(Dk) 
                return Out 
        else:
            print("This is not an orthogonal matrix or you ask for too many eigenvalues.")
            return 
    
    def truncate_mat(self, A_mat, k_princomps): # Αυτή η συνάρτιση θα χρησιμοποιηθεί για τον εκφυλισμό των 4 καναλιών κάθε εικόνας που θέλουμε να συμπιέσουμε
            if k_princomps > A_mat.shape[0]:
                print("__We cannot have more principal components that dimencions__")
                sys.exit()
                
            else:
                U, S, Vt = np.linalg.svd(A_mat, full_matrices=False) # Foull matrices helps if we have squared matreces
                # Truncate the matrices
                Uk = U[:, :k_princomps]
                Sk = np.diag(S[:k_princomps])
                Vt_k = Vt[:k_princomps, :]

                # REconstruct Image
                compressed_chann = Uk @ np.dot(Sk, Vt_k) # Ο ανακατασκευασμένος πίνακας τάξης κ
                return compressed_chann, Sk
        
    def compress_image(self, Image, k_princomps=50, Plot=True):
        print(f"__Compress using {k_princomps}__" ) 
        Channels = []
        for channel in Image:
            # print(f"Channel shape = {channel.shape}")
            # print(k_princomps) 
            trunc_channel, Sk = self.truncate_mat(channel, k_princomps)
            
            # Update the original channel with the compressed channel
            channel[:, :] = trunc_channel
            Channels.append(trunc_channel) 
        # Merge color channels back into the image
        print(Sk) 
        compressed_img_array = np.stack(Channels, axis=-1)    # stack = Join a sequence of arrays along a new axis.
        if Plot:
            self.plot_single_image(compressed_img_array,title=f"Using {k_princomps} principal components") 
            plt.clf() 
        # θέλουμε επίσης να υπολογίσουμε το σφάλμα σε σχέση με την αρχική εικόνα: 
        Image = np.array(Image)
        Image = np.reshape(Image, compressed_img_array.shape)

        error = np.linalg.norm(compressed_img_array - Image)**1/2 
        return compressed_img_array , error # θα επρεπε να χρησιμοποιήσω αυτό το σφάλμα για να κάνω ενα γράφημα
    
    def plot_single_image(self, image_matrix, title="4ch pic"):
        if image_matrix.ndim != 3 or image_matrix.shape[-1] != 4:
            raise ValueError("Input matrix should be 3D with shape (height, width, 4)")

        plt.imshow(image_matrix)
        plt.title(title)
        plt.axis('off')
        plt.show()

    def plot_errors(self,errors): 
        """Assuming Errors is a list of tuples (k, error)"""
        ks, errors = zip(*errors)

        # Plot the errors
        plt.plot(ks, errors, marker='o')
        plt.xlabel('k')
        plt.ylabel('Euclidean Norm of Error')
        plt.title('Reconstruction Error vs. k')
        plt.grid()
        plt.show()
        plt.clf() 

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
        def create_A(): # διαδικασία κατασκευής Α
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
        def check_if_Markov(M):
            column_sums_close_to_1 = np.allclose(np.sum(M, axis=0), 1.0)
            if column_sums_close_to_1:
                return True
            else:
                return False
        n = len(list(graph_data.keys()))
        d = .85
        print(f"We have a {n}*{n} matrix.")
        B = np.ones((n,n)) 
        A = np.array(create_A()).T 
        # print(A,"\n")  
        M = d*A + ((1-d)/n)*B
        if check_if_Markov(A) and check_if_Markov(M):
            # print(M) 
            print("__Markov created__")
            return M
        else:
            print("wrong")
            return 

    def find_important_node(self, graph_data):
        M = self.createMarkov(graph_data)
        eigen_dict = self.power_method(M) 

        # Get the indices of the greatest values in descending order
        main_eigenvec = eigen_dict[next(iter(eigen_dict))]
        indices_of_greatest_values = np.argsort(main_eigenvec)[::-1]
        return indices_of_greatest_values+1




class Ploting:
    def __init__(self) -> None:
        pass 
        
    def make_graph(self, graph_data):
        graph_data = self.convert_to_int_labels(graph_data)
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

    def convert_to_int_labels(self, graph_data):
        int_graph_data = {}

        for node, neighbors in graph_data.items():
            int_node = int(node)
            int_neighbors = [int(neighbor) for neighbor in neighbors]
            int_graph_data[int_node] = int_neighbors

        return int_graph_data
















erg_inst = Erg1_TemMl("/home/thinpan/Desktop/24/ml tem/Ask/1erg/project_finals") 
images_paths = ["/home/thinpan/Desktop/24/ml tem/Ask/1erg/project_finals/input/fotos/python_logo.png",
                "/home/thinpan/Desktop/24/ml tem/Ask/1erg/project_finals/input/fotos/uoc_logo.png"] 
pylogo = erg_inst.load_image_to_matrix(images_paths[0])
uoclogo = erg_inst.load_image_to_matrix(images_paths[1]) 

# distinct the channels of each picture 
def boring_proc(images, names):
    """Epistrefei ta channels ton taniston"""
    Channels = {}
    for i, img_array in enumerate(images):
        red_channel = np.array(img_array[:, :, 0]  )
        green_channel =np.array( img_array[:, :, 1]  )
        blue_channel = np.array(img_array[:, :, 2]  )
        alpha_channel = np.array(img_array[:, :, 3]  ) # όλοι  (203, 601)
        Channels[names[i]] = [red_channel, green_channel, blue_channel, alpha_channel]
    return Channels 
# names = ["pylogo","uoclogo"]
# images = [pylogo, uoclogo]
# Channels = boring_proc(images, names) 
# print(Channels["pylogo"][1].shape) 

# Images, Errors = [],[]
# for k in range(1,201,10):
#     compressed_Img, error = erg_inst.compress_image(Channels["uoclogo"],k_princomps=k)    
#     Images.append(compressed_Img) 
#     Errors.append((k,error)) 
# erg_inst.plot_errors(Errors) 

# json_graph1 = "/home/thinpan/Desktop/24/ml tem/Ask/1erg/project_finals/input/graphs/graph1.json" 
# with open(json_graph1, 'r') as file:
#     # Load the JSON content
#     graph_data = json.load(file)
# print(graph_data)    
# M = erg_inst.createMarkov(graph_data) 
# n = M.shape[0] 
# x0 = np.array([1/n for n in range(1,M.shape[0]+1)]) 
# eigen_dict = erg_inst.power_method(M,x0,n_eigonvalues=n)  
# # print(eigen_dict) 
# # eigenvalues, eigenvectors = np.linalg.eig(M) 
# eigenvalues, eigenvectors = list(eigen_dict.keys()),list(eigen_dict.values() )
# print(eigenvalues)  
# # Get the indices of eigenvalues in descending order of importance
# sorted_indices = np.argsort(eigenvalues)[::-1]
# for i, index in enumerate(sorted_indices):
#     print(index+1)  

