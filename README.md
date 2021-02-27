# data_analysis_4S_manuscript
Jupyter notebooks and dataset for Sewage, Salt, Silica and SARS-CoV-2 (4S): An economical kit-free method for direct capture of SARS-CoV-2 RNA from wastewater.

Authors: Oscar N. Whitney, Lauren C. Kennedy, Vinson Fan, Adrian Hinkle, Rose Kantor, Hannah Greenwald, Alexander Crits-Christoph, Basem Al-Shayeb, Mira Chaplin, Anna C. Maurer, Robert Tjian, Kara L. Nelson

Two jupyter notebooks are included:
1. 4S_data_analysis_RT_qPCR.ipynb (fully executable)
2. 4S_data_analysis_geopandas_and_clinical_data.ipynb (code only -- sewershed map not included)

Each notebook uses the same base files
1. qPCR_plate_info.csv (information about each  qPCR plate that was run)
2. qPCR_raw_data.csv (data from the QuantStudio 3)
3. Sample_inventory_conc_extract.csv (metadata about each wastewater sample and concentration/extraction details)
4. reprocess_qpcr.py
5. calculations.py

The 4S_data_analysis_geopandas_and_clinical_data.ipynb notebook has additional files
1. GIS Data (publicly available shapefiles for counties in CA and zip codes in CA) -- missing internal sewershed map
2. Case data (publicly  available clinical COVID-19 data from Alameda and Contra Costa Counties about COVID-19 cases
