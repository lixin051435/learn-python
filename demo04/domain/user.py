class User:
    def __init__(self,username,password):
        self.username = username
        self.password = password

    def __str__(self):
        return "[username:%s,password:%s]"%(self.username,self.password)
