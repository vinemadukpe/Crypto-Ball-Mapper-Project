# Crypto-Ball-Mapper-Project
Optimizing Cryptocurrencies Portfolio Using Ball Mapper
## 40 Cryptocurrencies Closing Prices Dataset for 2020 to 2024
`data = pd.read_csv('BM_COINS_Main.csv')`
`data.head()`
![image](https://github.com/user-attachments/assets/7bb3b631-da46-4449-a7a9-2c46f78312ce)
## Visualization
`import numpy as np`
`from scipy.spatial.distance import pdist, squareform`
`import networkx as nx`
`import matplotlib.pyplot as plt`
`import warnings`
`warnings.filterwarnings("ignore")`
`import pandas as pd`

`data = pd.read_csv('BM_COINS_Main.csv')`
`def Ballmapper(data, region_col, eps, metric='cosine'):`
    # Extracting numerical data and region information
   ` numerical_data = data.drop(columns=['Ticker', region_col])`
   ` regions = data[region_col].valACues`
    
    # Computing pairwise distances based on correlation
  `  dist_matrix = squareform(pdist(numerical_data, metric))`
   ` n_samples = dist_matrix.shape[0]`
    
    # Initializing graph and landmark selection
    bm_graph = nx.Graph()
    landmarks = []
    points_covered = set()
    
    # Landmark selection and cluster formation
    while len(points_covered) < n_samples:
        for i in range(n_samples):
            if i not in points_covered:
                landmarks.append(i)
                points_covered.add(i)
                # Create a node for each landmark
                bm_graph.add_node(i, region=regions[i], points_covered=[i])
                break
        # Expanding clusters around landmarks
        for j in range(n_samples):
            if dist_matrix[landmarks[-1], j] <= eps:
                points_covered.add(j)
                bm_graph.nodes[landmarks[-1]]['points_covered'].append(j)
    
    # Connecting clusters with overlapping points
    for i in range(len(landmarks)):
        for j in range(i + 1, len(landmarks)):
            if set(bm_graph.nodes[landmarks[i]]['points_covered']) & set(bm_graph.nodes[landmarks[j]]['points_covered']):
                bm_graph.add_edge(landmarks[i], landmarks[j])
    
    return bm_graph, regions



`eps = 0.1`

#### Generating the Ball Mapper graph with specified parameters
`bm_graph, regions = Ballmapper(data, 'Category', eps, metric='cosine')`

#### Define unique regions and assign a base color map
`unique_regions = np.unique(regions)
color_map = plt.cm.get_cmap('hsv', len(unique_regions))`

#### Customization of colors for specific regions
`region_to_color = {region: color_map(i) for i, region in enumerate(unique_regions)}
region_to_color['Platform Coin'] = '#a6d96a'
region_to_color['Blue-Chip Coin'] = '#14AAF5'
region_to_color['Stable Coin'] = '#03c04a'
region_to_color['Other Coin'] =  '#FAD000'`
#### Determine node sizes based on the number of points in each cluster
`node_sizes = [len(bm_graph.nodes[i]['points_covered']) * 100 for i in bm_graph.nodes]  # Adjust size factor as needed`

#### Node colors based on the region-to-color mapping
`node_colors = [region_to_color[bm_graph.nodes[i]['region']] for i in bm_graph.nodes]`

#### Visualization
`plt.figure(figsize=(6, 5))
pos = nx.spring_layout(bm_graph, k=0.4, iterations=5, seed=47)
nx.draw_networkx_edges(bm_graph, pos, alpha=0.5)
nx.draw_networkx_nodes(bm_graph, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)
nx.draw_networkx_labels(bm_graph, pos, font_size=8, horizontalalignment='center')`

#### Legend to reflect colors
`legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=region) for region, color in region_to_color.items()]
plt.legend(handles=legend_handles, title="Crypto Categories", bbox_to_anchor=(1, 1))
#plt.title("PM2.5 Ball Mapper Graph with Cosine Metric, eps=0.065")
plt.axis('off')
plt.tight_layout()
plt.savefig('BM_Coins_main.jpeg', dpi=1000)
plt.show()`

![image](https://github.com/user-attachments/assets/eff3f592-f5b5-4a23-abe3-dfbfb5307238)
## Sve Node Details
`import pandas as pd`

`def node_details(G, data):`
#### Using list comprehension to gather node details directly
   ` node_details_list = [{
        'Node ID': node,
        'Category': G.nodes[node]['region'],
        'Points Covered': len(G.nodes[node]['points_covered']),
        'Ticker': ", ".join(data.iloc[G.nodes[node]['points_covered']]['Ticker'].astype(str).tolist())
    } for node in G.nodes()]`
    
    #### Converting the list of dictionaries to a DataFrame
  `  node_details_df = pd.DataFrame(node_details_list)
    
    return node_details_df`

#### function with the graph object and data, and store the result in a variable
`node_details_df = node_details(bm_graph, data)`

#### Now you can directly work with the DataFrame in Python
`print(node_details_df.head())  # For example, print the first few rows`

#### save the DataFrame to a CSV file
`node_details_df.to_csv('BM_Coins_main_Nodes.csv', index=False)`

