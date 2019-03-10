import logging

# root logger, wont be overwritten if other modules are imported
# get a logger for each module
#logging.basicConfig(level=logging.INFO,  # CamelCase, printing debug statements
#                    format='[%(asctime)-8s] [%(name)-8s] [%(levelname)-1s] [%(message)s]')


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('[%(asctime)-8s] [%(name)-8s] [%(levelname)-1s] [%(message)s]')

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

#file_handler = logging.FileHandler('logger.txt')
#file_handler.setFormatter(formatter)
#file_handler.setLevel(logging.ERROR)
#logger.addHandler(file_handler)



class Employee(object):

    def __init__(self, first, last):
        self.first = first
        self.last = last
        self.__email =""
        logger.info("Created Employee: {} {}".format(first, last))

    @property
    def email(self):
        return self.__email

    @email.setter
    def email(self, val):
        self.__email = val

    @property
    def fullname(self):
        return self.first+self.last

if __name__ == "__main__":
    employee = Employee("John", "Wick")
    print(employee.email)
    employee.email = "lol@rofl.com"
    print(employee.email)
