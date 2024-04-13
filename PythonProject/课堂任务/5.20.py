'''class Person:
    __count = 0
    def __init__(self, name, gender, birth_day):
        self.name = name
        self.__gender = gender
        self.__birth_day = birth_day
        Person.__count += 1

    def info(self):
        print(f'姓名：{self.name}')
        print(f'性别：{self.__gender}')
        print(f'出生日期：{self.__birth_day}')


x = Person('Mary', 'F', '2049-08-01')
x.info()
print(x.name)'''

class User:
    def __init__(self, name, password = 6 * '1'):
        self.name = name
        self.password = password

    def info(self):
        print(f'{self.name}:{self.password}')

    def modify_password(self, new_password):
        self.password = new_password


class Admin(User):
    def __init__(self, name='admin000', password = 6 * '1'):
        super().__init__(name, password)

    def ban_user(self, u):
        print(f'封禁帐号{u.name}')
        del u.name

u1 = User('xyz', '123')
u1.info()

admin1 = Admin()
admin1.info()
admin1.ban_user(u1)
try:
    u1.info()
except:
    print('Founded 404')
