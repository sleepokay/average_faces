import requests, json, urllib
        

for i in range(4, 36):
    r = requests.get('https://www.recurse.com/api/v1/batches/' + str(i) + '/people/?access_token=<insert access token here')
    batch = r.json()

    print(type(batch))
    for j, entry in enumerate(batch):

        try:
            if entry['has_photo']:
                image_url = entry['image']
                urllib.request.urlretrieve(image_url, "faces/" + str(i).zfill(2) + "/" + str(j+1).zfill(2) + ".jpg")
        except TypeError:
            print(entry)
