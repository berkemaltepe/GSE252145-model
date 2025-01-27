# GSE252145-model
Build and run the docker container using:
`docker compose up`

Python Dash visualizations can be accessed at http://localhost:8050/ after running container.

## Data Processing
First, the data is downloaded from [the NCBI Website](https://www.ncbi.nlm.nih.gov/geo/download/?type=rnaseq_counts&acc=GSE252145&format=file&file=GSE252145_norm_counts_FPKM_GRCh38.p13_NCBI.tsv.gz) using pandas. Then the data is transposed and cleaned up to create a dataset for training.

The labels (pre-treatment, post-treatment) are not included in the data itself, and is provided on the [information page](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE252145) of the GSE252145 dataset.

The data had a very high number of feautures (39376) compared to the sample size (31), so to avoid overfitting I used the PCA algorithm to reduce the dimensionality of the data. The resulting explianed variance ratio is 1, which means that the dimensionality reduction captures all of the variance of the original feature set.

## Neural Net and Results
Since the problem is a fairly linear binary classification problem, the neural architecture is very simple, consisting of three fully connected layers.

Since the sample size was very low, the average accuracy of the model was only around 50%, which is the same as random guessing for a binary classification problem.