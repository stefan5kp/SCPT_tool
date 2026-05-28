# SCPT_tool

This tool is designed to automate the Vs interpretation process from SCPT. As of May 15, 2026, the direct interpretation tool is available in the folder with sample files. All-in-one tool is a comprehensive tool set for SCPT - Vs interpretation or Vs estimate from CPT data. The tool is currently at beta stage, more features will be added to the tool at its official release.  

## **User Guide**  
    
1. If you wanted to purely interpret the traveltime data into stepped Vs profiles from SCPT, please use the direct interpretation tool  
2. If you wanted to estimate Vs profiles from purely cone penetration data, please use the national model tool  
3. Site-specific joint algorithm that leverage both the traveltime and cone penetration data is coming soon  


## Update Log 
    
2026-05-19  
	- Correction Made to the Robertson 2012 Calculation in the all-in-one tool.  
	- National Model Tool is now available with the most recent version of the model and the corrected Robertson 2012 Calculation. 

2026-06-01  
    - Updated input parameters for the National Model to be inverse-filtered qt and fs (Boulanger and DeJong, 2018).  
    - Added Kc plot and shading for Ic > 2.6 for National Model.  
    - Updated name to be Zhang et al. model.  
    - Updated the cost function for the direct interpretation tool.
