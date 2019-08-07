import glob

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

def create_text(imgs, ndict):
    texts = []
    for img in imgs:
        texts.append(ndict[img])

    return texts

def main():
    ncol = 6
    imgs = glob.glob('*.png')
    img_cells = [img_cell(v) for v in imgs]
    ndict = read_names('input.txt')
    texts = create_text(imgs, ndict)
    text_cells = [text_cell(v) for v in texts]
    merged_cells = merge_cells(img_cells, text_cells, ncol)
    html = make_table(merged_cells, ncol)
    with open('index.html', 'w') as f:
        f.write(html)

    
if __name__ == '__main__':
    main()
