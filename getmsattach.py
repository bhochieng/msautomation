#install msal via $pip install msal
from msal import ClientApplication

class AttachmentDownloader:
    def __init__(self, username: str, password: str):
        self.client_id = '<your client id>'
        self.authority = 'https://login.microsoftonline.com/bhochieng'

# Initialise MS ClientApplication object with your client_id and authority URL
        self.app = ClientApplication(client_id=self.client_id,authority=self.authority)
        self.username = username # your mailbox username
        self.password = password # your mailbox password

if __name__ == "__main__":
    downloader = AttachmentDownloader("zzz@outlook.com", "password")
#Now that we have our app object initialised, we can acquire the token. This token can be used to extract to access_token for headers.
    token = self.app.acquire_token_by_username_password(username=self.username,password=self.password,scopes=['.default'])
    print(token)
