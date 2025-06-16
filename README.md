# h5ad_viewer
Tool for interactively viewing h5ad objects using streamlit

To use the tool
1. set up a python environment using the requirements.txt provided here using `pip install -r requirements.txt`
2. from the command line run: streamlit run h5ad_viewer.py
   a. if you intend to run the viewer on files larger than 200MB (streamlit default size limit) add `--server.maxUploadSize XX` where XX is larger than the file you want to load
   EX: if you wanted to view a 260 MB file you would run streamlit run h5ad_viewer.py --server.maxUploadSize 300.
3. You should then see a browser window open with a box for uploading an h5ad file and a basic interface for visualizing UMAP, PCA, and Gene Expression data.
   
