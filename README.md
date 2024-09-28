These files include my code from my project on predicting the susceptibility of protein sites to oxidation at the University of Copenhagen. This project spanned ~2.5 months, from June to August. I intend to use ESM embeddings for the equivariant graph neural network when the semester cools down a bit :)  

data_processing: Clean dataset to ensure that it matches the information in PDB files. Extract corresponding PDB files from RSCB, then extract relevant alpha carbon coordinates and residue information. Format so that PDB information can be passed into ProteinMPNN and ESM models.  

egnn: Contains eGNN model and training, test, validation code. Not in repository since I adapted proprietary code from my lab. 

generate_embeddings: Use ProteinMPNN & ESM to extract respective embeddings.   

logistic_regression: Simple logistic regression task for testing the capability of ESM embeddings to predict susceptibility to protein oxidation  

RSC758, BALOSCTdb: The 2 datasets I used for this project. Unfortunately, most of the lines of RSC758 were not used as there was minimal correspondence with the PDB files.
