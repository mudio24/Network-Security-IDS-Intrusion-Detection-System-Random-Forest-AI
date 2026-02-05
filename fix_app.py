import re

with open(r'c:\laragon\www\soc-streamlite\app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    
    # Find and fix the problematic section (around line 417-425)
    if "df_display = df_connections[display_columns].copy()" in line and i > 400:
        new_lines.append(line)  # Keep original copy line
        i += 1
        
        # Skip old column rename line (if it exists) and add our fix
        if i < len(lines) and "df_display.columns" in lines[i]:
            # Add format confidence first
            new_lines.append("        df_display['confidence'] = df_display['confidence'].apply(lambda x: f\"{x:.2%}\")\n")
            # Add status_col variable
            new_lines.append("        status_col = t('status')\n")
            # Add column rename with status_col variable
            new_lines.append("        df_display.columns = [t('source'), t('destination'), t('protocol'), t('bytes'), t('count'), status_col, t('confidence')]\n")
            i += 1  # Skip old df_display.columns line
            
            # Skip old Confidence formatting line if exists  
            if i < len(lines) and "df_display['Confidence']" in lines[i]:
                i += 1
            
            # Copy blank line and comments
            while i < len(lines) and (lines[i].strip() == "" or lines[i].strip().startswith("#")):
                new_lines.append(lines[i])
                i += 1
            
            # Fix highlight_attacks function
            if i < len(lines) and "def highlight_attacks" in lines[i]:
                new_lines.append(lines[i])  # def line
                i += 1
                # Fix the if condition to use status_col
                if i < len(lines) and "if row['Status']" in lines[i]:
                    new_lines.append("            if row[status_col] == 'Attack':\n")
                    i += 1
                elif i < len(lines) and "if row[" in lines[i]:
                    new_lines.append(lines[i])  # Already fixed
                    i += 1
        continue
    
    new_lines.append(line)
    i += 1

with open(r'c:\laragon\www\soc-streamlite\app.py', 'w', encoding='utf-8', newline='') as f:
    f.writelines(new_lines)

print('Fixed successfully!')
