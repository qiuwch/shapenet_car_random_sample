import glob

# Settings 
model_name = "resnet"
part_name = 'fl'
filt_name = 'focus'
input_dir = '/home/xwei18/shapenet_car/htmls/sigmoid/{}_ft_{}_same.txt'.format(model_name, part_name)

iffilter = True
filt_dir = '/home/xwei18/shapenet_car/focus_names/{}_ft_{}_same/{}.txt'.format(model_name, part_name, filt_name)

def read_filter(path):
    file = open(path, 'r')
    content_list = file.readlines()
    flist = []
    sdict = {}
    for line in content_list:
        content = line.strip().split(' ')
        flist.append(content[0])
        sdict[content[0]] = content[1]
        
    return flist, sdict

def img_cell(content):
    return '<td><img src="%s" width="200px"/></td>' % content

def text_cell(content):
    return '<td>%s</td>' % content
    
def make_table(cells, ncol):
    table = ''
    for i, cell in enumerate(cells):
        if i % ncol == 0:
            table += '<tr>'
        table += cell
        if i % ncol == ncol - 1:
            table += '</tr>'

    return '<table>%s</table>' % table

def merge_cells(cells1, cells2, ncol):
    merged = []
    i = 0
    cells1 += [''] * (ncol - len(cells1) % ncol)
    cells2 += [''] * (ncol - len(cells2) % ncol)
    while len(merged) != len(cells1) + len(cells2):
        merged += cells1[i*ncol:(i+1)*ncol]
        merged += cells2[i*ncol:(i+1)*ncol]
        i+=1
    return merged

def read_names(path):
    file = open(path, 'r')
    content_list = file.readlines()
    ndict = {}
    for line in content_list:
        content = line.strip().split(' ')
        ndict[content[0]] = content[1]+' '+content[2]
    
    return ndict

def create_text(imgs, ndict, flist=None, sdict=None):
    texts = []
    if flist == None:
        for img in imgs:
            texts.append(ndict[img])
    else:
        for img in imgs:
            if img in flist:
                texts.append([img, ndict[img], sdict[img]])

    return texts

def main():
    ncol = 4
    imgs = glob.glob('*.png')
    if iffilter:
        flist, sdict = read_filter(filt_dir)
        img_cells = []
        for v in imgs:
            if v in flist:
                img_cells.append(img_cell(v))
    else:
        img_cells = [img_cell(v) for v in imgs]
    ndict = read_names(input_dir)
    if iffilter:
        texts = create_text(imgs, ndict, flist, sdict)
    else:
        texts = create_text(imgs, ndict)
    text_cells = [text_cell(v) for v in texts]
    merged_cells = merge_cells(img_cells, text_cells, ncol)
    html = make_table(merged_cells, ncol)
    with open('index.html', 'w') as f:
        f.write(html)

    
if __name__ == '__main__':
    main()
