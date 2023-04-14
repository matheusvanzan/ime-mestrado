import requests



def send(text):
    # print('Whats: sending text')
    # print(text)
    # %20 for space, %0A for new lines
    recipients = [
        ('+5521996740135', '198079'),
    ]

    try:
        mask = 'https://api.callmebot.com/whatsapp.php?phone={phone}&text={text}&apikey={key}'
        for phone, key in recipients:
            text = text.replace('%', '%25')
            text = text.replace(' ', '%20')
            text = text.replace('\n', '%0A')
            response = requests.get(mask.format(phone=phone, text=text, key=key))
            # print(f'Whats: message sent to {phone}')
    except Exception as e:
        print(e)