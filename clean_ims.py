from google_images_search import GoogleImagesSearch

# if you don't enter api key and cx, the package will try to search
# them from environment variables GCS_DEVELOPER_KEY and GCS_CX
gis = GoogleImagesSearch('AIzaSyAOffuAl5cbWBEPSIiL8QZQOHQ_agzuohs', '014542844488610288047:kje7jujvoi0')

# example: GoogleImagesSearch('ABcDeFGhiJKLmnopqweRty5asdfghGfdSaS4abC', '012345678987654321012:abcde_fghij')

#define search params:
_search_params = {
    'q': 'protoplanetarydisks',
    'num': 50,
    'safe': 'medium',
    'fileType': 'jpg',
    'imgSize': 'medium'
}

# this will only search for images:
#gis.search(search_params=_search_params)

# this will search and download:
#gis.search(search_params=_search_params, path_to_dir='/path/')

# this will search, download and resize:
#gis.search(search_params=_search_params, path_to_dir='/path/', width=500, height=500)

# search first, then download and resize afterwards
gis.search(search_params=_search_params)
for image in gis.results('/Users/donut/Desktop/images/'):
    image.download()
    image.resize(128, 128)