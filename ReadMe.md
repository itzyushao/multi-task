# 安裝dependency 

pip install pytorch_lightning 

# 如何執行訓練: 

python train_avm_module.py

# 檔案說明: 

--- 

1. train_avm_module.py: 用來執行training 

2. avm_modules.py: 
   - AVMProjModule: 繼承了pl.LightningModule的套件，定義了訓練以及驗證時的邏輯、資料前處理、LOG的儲存、以及最佳化方法。
   - GraphNeuralNet、GCNConv: 定義了Network架構。

3. avm_etl_module.py: 內含資料前處理的邏輯。 
   - get_geometric_data: 主要的前處理程序。
   - load_raw_cities_df: 載入原始資料用。
   - convert_lat_lon_to_radian、get_city_dist_matrix、convert_to_networkx_graph、convert_to_geometric_data: 資料轉換程序。
