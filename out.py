def get_output(rows, staff_lines, avg_spacing, out_path, img_name):
    heights = ['b2', 'a2', 'g2', 'f2', 'e2', 'd2', 'c2', 'b1', 'a1', 'g1', 'f1', 'e1', 'd1', 'c1']
    notes = {
        'chord': '/4',
        'a_1': '/1',
        'a_2': '/2',
        'a_4': '/4',
        'a_8': '/8',
        'a_16': '/16',
        'a_32': '/32',
        'b_8': '/8',
        'b_16': '/16', 
        'b_32': '/32'
    }
    ignored = ['barline', 'clef']
    accidentals = {
        'dot':'.',
        'double_flat': '&&',
        'double_sharp': '##',
        'flat': '&',
        'natural': '',
        'sharp': '#',
    }
    meters = {
        "t_4_2": "4\\2", 
        "t_4_4": "4\\4"
    }
    img_name = img_name.split('.')[0]
    # neglecting image extension
    # constructing output file path
    with open(out_path + '/' + img_name + '.txt', 'w+') as f:

        # variable to store accidentals 
        # accidentals comes before notes 
        # but in the output file 
        # they need to be after the note
        accidental = ''
        
        f.write('{\n')
        
        # loop for each row in the image
        for k in range(len(rows)):
            if (k%2 == 0):
                f.write('[\\meter<"')

            # loop for each object detected in the row
            for symbol in rows[k]:
                
                if symbol['label'] in meters.keys():
                    f.write(meters[symbol['label']] + '"> ')
                    continue

                elif symbol['label'] in ignored:
                    continue
                
                elif symbol['label'] in accidentals.keys():
                    accidental += accidentals[symbol['label']]
                    continue
                
                else:
                    notes_to_write = []
                    for center in symbol['centers']:
                        note = ''
                        if(staff_lines[0] < center < staff_lines[6]):
                            i=0
                            # loop to detect which 2 lines contain the center
                            while(center > staff_lines[i+1]):
                                i += 1

                            # center is closer to the upper line
                            if center <= (staff_lines[i] + (0.1 * avg_spacing)):
                                note += (heights[ 2*i + 1 ] + notes[symbol['label']])
                            
                            # center is closer to the lower line
                            elif center >= (staff_lines[i+1] - (0.1 * avg_spacing)):
                                note += (heights[ 2*(i+1) + 1 ] + notes[symbol['label']])
                            
                            # center is closer to the middle
                            else:
                                note += (heights[2*i + 2] + notes[symbol['label']])
                        
                        # center is greater than the last staff line 
                        # then it must be a c1
                        elif center >= staff_lines[6]:
                            note += (heights[13] + notes[symbol['label']])
                        
                        # center is closer to the first staff line
                        else:
                            if center >= (staff_lines[0] - (0.1 * avg_spacing)):
                                note += (heights[1] + notes[symbol['label']])
                            else:
                                note += (heights[0] + notes[symbol['label']])
                        if accidental != '':
                            notes_to_write.append(note)
                        else:
                            notes_to_write.append(note + ' ')
                    
                    if(symbol['label'] == 'chord'):
                        f.write('{')

                        # to stick to the alphabetical order
                        notes_to_write = sorted(notes_to_write) 

                    for index, note in enumerate(notes_to_write):
                        if accidental != '' and not (accidental in ['.', '..']):
                            f.write(note[0]+accidental+note[1])
                            accidental = ''
                        else:
                            f.write(note)
                        if (index != (len(notes_to_write)-1) and symbol['label'] == 'chord'):
                            f.write(', ')
                    
                    if(symbol['label'] == 'chord'):
                        f.write('}')

                # check if there was an accidental 
                if accidental != '':
                    f.write(accidental + ' ')
                    accidental = ''

            # check if it's the last row
            if((k%2==1)):
                if k != (len(rows) - 1):
                    f.write('],\n')
                else:
                    f.write(']\n')
        f.write('}')












# =====================================================================================
# =====================================================================================
# ===================================== TESTING =======================================
# =====================================================================================
# =====================================================================================


# 'b2', 7
# 'a2', 10
# 'g2', 13
# 'f2', 15.1
# 'e2', 19.7
# 'd2', 19.8
# 'c2', 21
# 'b1', 25
# 'a1', 26
# 'g1', 30.1
# 'f1', 31
# 'e1', 35
# 'd1', 37
# 'c1', 42

# staff = [10, 15, 20, 25, 30, 35, 40]
# avg = 5
# to_check = [
#     [
#         {
#             'label': 't_4_4',
#             'centers': [7]
#         },
#         {
#             'label': 'a_1',
#             'centers': [7]
#         },
#         {
#             'label': 'a_2',
#             'centers': [10]
#         },
#         {
#             'label': 'double_flat',
#             'centers': []
#         },
#         {
#             'label': 'a_4',
#             'centers': [13]
#         },
#         {
#             'label': 'sharp',
#             'centers': []
#         },
#         {
#             'label': 'a_8',
#             'centers': [15.1]
#         },
#         {
#             'label': 'flat',
#             'centers': []
#         },
#     ],
#     [
#         {
#             'label': 'a_16',
#             'centers': [19.4]
#         },
#         {
#             'label': 'double_sharp',
#             'centers': []
#         },
#         {
#             'label': 'a_32',
#             'centers': [19.8]
#         },
#         {
#             'label': 'b_8',
#             'centers': [21]
#         },
#         {
#             'label': 'b_16',
#             'centers': [25]
#         },
#         {
#             'label': 'b_32',
#             'centers': [26]
#         },
#         {
#             'label': 'chord',
#             'centers': [30.1,31,35,37,42]
#         },
#     ]
# ]
# get_output(to_check, staff, 5, '/home/francois/Desktop', 'test.jpg')

