import os

def sort_tex_files(directory):
    tex_files = [f for f in os.listdir(directory) if f.endswith('.tex') and f.split('.', 1)[0] not in ['Appendix', 'Reference']]    
    modified_files = []
    tex_files.sort()
    
    for i, file_name in enumerate(tex_files, start=1):
        new_name = f"{i}.{file_name.split('.', 1)[1]}"
        os.rename(os.path.join(directory, file_name), os.path.join(directory, new_name))
        modified_files.append((file_name, new_name))
        print(f"Renamed {file_name} to {new_name}")
    modified_files.append(('Appendix.tex', 'Appendix.tex'))
    modified_files.append(('Reference.tex', 'Reference.tex'))
    print(modified_files)
    return modified_files

def update_main_tex(directory, modified_files):
    parent_directory = os.path.dirname(directory)
    main_tex_path = os.path.join(parent_directory, 'main.tex')
    
    with open(main_tex_path, 'r') as file:
        lines = file.readlines()
    
    new_lines = []
    count = 0
    appendix_found = False
    reference_found = False
    for line in lines:
        if 'Appendix.tex' in line:
            appendix_found = True
        if 'Reference.tex' in line:
            reference_found = True

        if line.startswith('\\input'):
            if count < len(modified_files):
                new_lines.append(f'\\input{{sections/{modified_files[count][1]}}}\n')
                count += 1
        
        elif line.startswith('\\end{document}') and not appendix_found and not reference_found:
            new_lines.append('\\input{sections/Appendix.tex}\n')
            new_lines.append('\\input{sections/Reference.tex}\n')
            new_lines.append(line)
        else:
            new_lines.append(line)

    # new_lines.append('\\input{sections/Appendix.tex}\n')
    # new_lines.append('\\input{sections/Reference.tex}\n')
    
    with open(main_tex_path, 'w') as file:
        file.writelines(new_lines)
    
    print(f"Updated {main_tex_path} with new file names.")



directory = '/Users/tony/Desktop/Tony/College/大二上/MCM/Thesis/sections'
modified_files = sort_tex_files(directory)
update_main_tex(directory, modified_files)