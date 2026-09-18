import pandas as pd
import markdown
from io import StringIO

def process_pfmea_to_chunks(raw_markdown_text: str) -> str:
    """
    Converts mixed LlamaParse output to pure HTML, parses tables, 
    forward-fills merged cells, and builds unbreakable grouped text chunks.
    """
    # 1. Convert the ENTIRE document to pure HTML
    # This standardizes both Markdown tables and raw HTML tables into <table> tags
    html_content = markdown.markdown(raw_markdown_text, extensions=['tables'])
    
    try:
        # 2. Extract every table found in the HTML
        all_tables = pd.read_html(StringIO(html_content))
    except ValueError:
        # No tables found
        return raw_markdown_text

    structured_text = ""
    
    for df in all_tables:
        # Only process tables that look like PFMEA grids (usually 12 columns)
        if len(df.columns) >= 11: 
            
            # 3. Forward-fill the empty cells (fixes the merged row problem)
            cols_to_ffill = df.columns[0:5]
            df[cols_to_ffill] = df[cols_to_ffill].ffill()
            
            # 2. Fill any blank cells in the remaining columns with a safe string
            # This prevents missing data from throwing errors or shifting
            df.fillna("Not specified", inplace=True)
            
            # 4. Group by the parent hierarchy (Step, Function, Failure Mode)
            # df.columns[0]=Step, [1]=Function, [2]=Failure Mode
            grouped = df.groupby([df.columns[0], df.columns[1], df.columns[2]])
            
            for (step, function, mode), group in grouped:
                # Build the unbreakable chunk
                block = f"PROCESS STEP: {step}\n"
                block += f"FUNCTION/REQUIREMENT: {function}\n"
                block += f"POTENTIAL FAILURE MODE: {mode}\n"
                
                # Extract Effect and Severity (columns 3 and 4)
                effect = group.iloc[0, 3] 
                severity = group.iloc[0, 4] 
                block += f"POTENTIAL EFFECT(S): {effect}\n"
                block += f"SEVERITY (S): {severity}\n\n"
                
                block += "CAUSES AND CONTROLS:\n"
                
                # Iterate through all causes grouped under this failure mode
                for _, row in group.iterrows():
                    cause = row.iloc[5]
                    occurrence = row.iloc[6]
                    prev_control = row.iloc[7]
                    det_control = row.iloc[8]
                    detection = row.iloc[9]
                    ap = row.iloc[10]
                    rec_action = row.iloc[11]
                    
                    block += f"- Cause: {cause} (Occurrence: {occurrence})\n"
                    block += f"  - Prevention Control: {prev_control}\n"
                    block += f"  - Detection Control: {det_control} (Detection: {detection})\n"
                    block += f"  - Action Priority (AP): {ap}\n"
                    block += f"  - Recommended Action: {rec_action}\n\n"
                
                # 5. Append the strict boundary delimiter
                structured_text += block + "===CHUNK_BOUNDARY===\n\n"
        
        else:
            # For non-PFMEA tables, just convert back to markdown and append
            structured_text += df.to_markdown(index=False) + "\n\n===CHUNK_BOUNDARY===\n\n"

    return structured_text