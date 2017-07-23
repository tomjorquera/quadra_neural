import urllib, os, subprocess
from bs4 import BeautifulSoup

URL_BASE = 'https://www.laquadrature.net'
PATH_CPs = '/fr/espace-presse?page='
NB_PAGES = 33

SOURCE_DIR='./html'
OUT_DIR='./data'

# Download all CPs into `SOURCE_DIR`
for i in range(NB_PAGES + 1):
    response = urllib.request.urlopen(URL_BASE + PATH_CPs + str(i))

    # get all CP urls from the page
    CPs = (BeautifulSoup(response.read(), 'html.parser')
           .findAll('td', {'class':'views-field-title'}))
    for cp in CPs:
        cp_id = cp.find('a')['href']
        cp_soup = BeautifulSoup(urllib.request.urlopen(URL_BASE + cp_id).read(), 'html.parser')

        # extract title
        cp_title = cp_soup.find('h1', {'id': 'page-title'})
        cp_title.extract()

        # remove social widget
        social = cp_soup.find('div', {'class':'share-links-container'})
        if social:
            social.extract()

        # convert to text
        cp_str = str(cp_title) + '\n' + '\n'.join([str(x) for x in cp_soup.contents if x != '\n'])

        # save into file
        cp_file = SOURCE_DIR + '/' + cp_id + '.html'
        try:
            if not os.path.exists(os.path.dirname(cp_file)):
                os.makedirs(os.path.dirname(cp_file))
            with open(cp_file, 'w') as f:
                f.write(cp_str)
                print('downloaded ' + cp_id)
        except Exception as e:
            print('error trying to create ' + cp_file)
            print(e)

# Convert HTML files to plaintext
for root, directories, filenames in os.walk(SOURCE_DIR):
    for filename in filenames:
        fqfn = os.path.join(root, filename)

        # some files contain erroneous `</br/>` tags, remove them
        subprocess.run(['sed', '-i', 's|</br/>||g', fqfn])

        # make some soup
        with open(fqfn) as fp:
            cp_soup = BeautifulSoup(fp, 'html.parser')
            cp_content = cp_soup.find('div', {'class': 'content'})

            # remove social widget
            social = cp_content.find('div', {'class':'share-links-container'})
            if social:
                social.extract()

            # convert to text
            cp_str =cp_content.text

            # save into file
            cp_file = OUT_DIR + '/' + filename + '.txt'
            try:
                if not os.path.exists(os.path.dirname(cp_file)):
                    os.makedirs(os.path.dirname(cp_file))
                with open(cp_file, 'w') as f:
                    f.write(cp_str)
                    print('created ' + cp_file)
            except Exception as e:
                print('error trying to create ' + cp_file)
                print(e)
