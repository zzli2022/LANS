# LANS
Code  and Data for ACL2024 Findings paper: "[*LANS: A Layout-Aware Neural Solver for Plane Geometry Problem*](https://arxiv.org/abs/2311.16476)". We propose a layout-aware neural solver named LANS, integrated with two new modules: multimodal layout-aware pre-trained language module (MLA-PLM) and layout-aware fusion attention (LA-FA). MLA-PLM adopts structural-semantic pre-training (SSP) to implement global relationship modeling, and point-match pre-training (PMP) to achieve alignment between visual points and textual points. LA-FA employs a layout-aware attention mask to realize point-guided cross-modal fusion for further boosting layout awareness of LANS. Extensive experiments on datasets Geometry3K and PGPS9K validate the effectiveness of the layout-aware modules and superior problem-solving performance of our LANS solver, over existing symbolic and neural solvers. 

<div align=center>
	<img width="300" src="figures\overall_lans.png">
</div>
<div align=center>
	Figure 1. Overview of PGPSNet solver.
</div>

<div align=center>
	<img width="400" src="figures\multimodal_layout_aware_pretraining.png">
</div>
<div align=center>
	Figure 1. Overview of PGPSNet solver.
</div>


## PGPS9K Dataset
You could download the dataset from [Dataset Homepage](http://www.nlpr.ia.ac.cn/databases/CASIA-PGPS9K).
In default, unzip the dataset file to the fold `./datasets`.
<!--or [Google Drive](https://drive.google.com/file/d/1xIloVheqwYCOI6XA84Yh_djyaVlQgjOo/view?usp=drive_link)-->


## Multimodal Layout-Aware Pre-training & LANS Finetuning
code comming soon
<!-- ## Multimodal Layout-Aware Pre-training
code comming soon

## LANS Fintuneing
code comming soon

## Test
code comming soon -->
