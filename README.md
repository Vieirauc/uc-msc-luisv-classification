# CFG Dataset

Repository with the dataset of vulnerable functions, with the CFG (Control Flow Graph) as their features.

Each file in the [datasets](datasets) folder has the `.tar.gz` extension, and it compress a `.csv` files with the following data:

* Unique description of each function (sample)
* Label indicating if the function is vulnerable or not
* Degree of the CFG, i.e., the number of nodes
* Adjacency Matrix of each CFG
* Features of each node of the CFG

The features of each sample are:
* Outdegree of the node (i.e., number of edges leaving the node)
* Indegree of the node (i.e., number of edges arriving in the node)
* Number of statements of the node
* Number of statements with type "Numeric Constants" in the node (available from v0.2)
* Number of statements with type "Transfer Instructions" in the node (available from v0.2)
* Number of statements with type "Call Instructions" in the node (available from v0.2)
* Number of statements with type "Arithmetic Instructions" in the node (available from v0.2)
* Number of statements with type "Compare Instructions" in the node (available from v0.2)
* Number of statements with type "Mov Instructions" in the node (available from v0.2)
* Number of statements with type "Terminations Instructions" in the node (available from v0.2)
* Number of statements with type "Data Declaration Instructions" in the node (available from v0.2)

