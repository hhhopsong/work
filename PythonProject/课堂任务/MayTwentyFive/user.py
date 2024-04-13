class User:
    def info(self):
        print('User')


class Admin(User):
    def info(self):
        print('Admin')


def main():
    u1 = User()
    u1.info()

    admin1 = Admin()
    admin1.info()


if __name__ == '__main__':
    main()
