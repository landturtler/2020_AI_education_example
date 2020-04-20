class Calc:
    def __init__(self) : # 생성자 메서드, 인스턴스 객체를 생성할 때 자동으로 호출, 인스턴스 멤버 초기화
        print('생성자가 호출됨')
        self.a = 0
        self.b = 0
        self.c = 0
    
    def add(self,a,b):
        self.a = a
        self.b = b
        self.c = a + b
        return self.c

    def subtract(self,a,b):
        self.a = a
        self.b = b
        self.c = a - b
        return self.c

    def multiply(self,a,b):
        self.a = a
        self.b = b
        self.c = a * b
        return self.c

    def divide(self,a,b):
        self.a = a
        self.b = b
        self.c = a / b
        return self.c

print('나는 모듈입니다')

if __name__ == '__main__':
    m1 = Calc()
    print(m1.add(10,20))
    print(m1.subtract(30,10))





