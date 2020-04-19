# 클래스기초실습문제.py
# 1번
class Car:

    def __init__(self):
        print('생성자')
        self.car_name = '소나타'
        self.car_drv = '전륜'
        self.car_speed = 0
        self.car_direction = '앞쪽'
        self.car_fuel = '휘발유'
        self.car_state = '정상'

    def set_car_name(self, name):
        self.car_name = name
        print("차종이 [",self.car_name,"]으로 변경 되었습니다")

    def get_car_name(self):
        return self.car_name

    def set_car_drv(self, drv):
        self.car_drv = drv
        print("차의 구동 방식이 [", self.car_drv ,"]으로 변경 되었습니다")

    def get_car_drv(self):
        return self.car_drv

    def set_car_fuel(self, fuel):
        self.car_fuel = fuel
        print("차의 연료 방식이 [", self.car_fuel,"]로 변경 되었습니다")

    def get_car_fuel(self):
        return self.car_fuel

    def set_car_state(self,state):
        self.car_state = state
        print("차의 상태가 [",self.car_state, "]으로 변경 되었습니다")

    def get_car_state(self):
        return self.car_state

    def set_speed(self,speed):
        self.car_speed = speed
        print("자동차의 속력이 시속 [",self.car_speed,"]km 로 변경되었습니다")

    def get_speed(self):
        return self.car_speed

    def turn(self,direction):
        self.car_direction = direction
        print("자동차의 방향이 [",self.car_direction ,"]으로 변경되었습니다")

    def stop(self):
        self.car_direction = '정지'
        print("자동차가 정지 하였습니다")

    def start(self):
        print("자동차가 시동이 걸렸습니다")

    def move_forward(self):
        self.car_direction = '앞쪽'
        print("자동차가 전진합니다 속도는 ",self.car_speed,"km입니다")

    def move_backward(self):
        self.car_direction = '뒤쪽'
        print("자동차가 후진합니다 속도는 ",self.car_speed,"km입니다")

    def __del__(self):
        print('[', self.car_name, "] 자동차가 제거되었습니다")


def test_car_class():
    sonata = Car()
    sonata.set_car_name('산타페')
    print(sonata.get_car_name())

    sonata.set_car_drv('4륜')
    print(sonata.get_car_drv())

    sonata.set_car_fuel('전기')
    print(sonata.get_car_fuel())

    sonata.set_car_state('브레이크고장')
    print(sonata.get_car_state())

    sonata.set_speed(100)
    print(sonata.get_speed())

    sonata.turn('오른쪽')
    sonata.stop()

    sonata.start()

    sonata.move_forward()

    sonata.move_backward()

    return sonata

# sonata = test_car_class()

# print('-'*30)

# 2번

class CarCenter:
    price = {'정상': 10, '브레이크고장': 1000, '전조등고장': 2000, '후미등고장': 3000, '연료부족': 4000,
            '타이어펑크': 5000, '엔진오일부족': 6000, '냉각수부족': 7000, '폐차처리': 9000}

    def __init__(self):
        self.fix_cost = 0
        self.fixed_list = {}
        # self.accent = Car()

    def fix_car(self,car):

        self.fix_cost = CarCenter.price[car.car_state]
        self.fixed_list[car.car_name] = car.car_state
        print('[',car.car_name,']의 [',car.car_state,
              '] 수리 완료, 비용은 [',self.fix_cost,'] 원 입니다')

    def set_car_drv(self,car, drv):
        car.car_drv = drv
        # self.accent.car_drv = drv
        print("차의 구동 방식이 [", car.car_drv ,"]으로 변경 되었습니다")

    def get_car_drv(self,car):
        return car.car_drv

    def set_car_fuel(self,car,fuel):
        car.car_fuel = fuel
        print("차의 연료 방식이 [", car.car_fuel,"]로 변경 되었습니다")

    def get_car_fuel(self,car):
        return car.car_fuel

    def get_fixed_list(self,car):
        fixed_item = self.fixed_list[car.car_name]
        cost = CarCenter.price[fixed_item]
        return '[' + fixed_item + '] : [' + str(cost) + ']원'

    def __del__(self):
        pass

def test_carcenter(car):
    sonata = car

    ct1 = CarCenter()

    ct1.fix_car(sonata)

    ct1.set_car_drv(sonata,'후륜')
    print(ct1.get_car_drv(sonata))

    ct1.set_car_fuel(sonata, '전기')
    print(ct1.get_car_fuel(sonata))

    print(ct1.get_fixed_list(sonata))


# test_carcenter(sonata)

# 별도의 파일로 작성한다
import 클래스기초실습문제 as car

avante = car.Car()
avante.set_car_name('아반테')
print(avante.get_car_name())
avante.set_car_state('전조등고장')
print(avante.car_state)

print('-'*30)
ct1 = car.CarCenter()
ct1.fix_car(avante)
ct1.set_car_drv(avante, '후륜')
print(ct1.get_car_drv(avante))

ct1.set_car_fuel(avante, '수소')
print(ct1.get_car_fuel(avante))
print(ct1.get_fixed_list(avante))


sorento = car.Car()
sorento.set_car_name('소렌토')
sorento.set_car_state('타이어펑크')
ct1.fix_car(sorento)
print(ct1.get_fixed_list(sorento))

pride = car.Car()
pride.set_car_name('프라이드')
pride.set_car_state('엔진오일부족')
ct1.fix_car(pride)
print(ct1.get_fixed_list(pride))

pride.set_car_state('타이어펑크')
ct1.fix_car(pride)
print(ct1.get_fixed_list(pride))

print(ct1.fixed_list)
# {'아반테': '전조등고장', '소렌토': '타이어펑크',
# '프라이드': '엔진오일부족'}
