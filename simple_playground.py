import math
import random as r
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from simple_geometry import *
import matplotlib.pyplot as plt
import matplotlib as mlt
from PyQt5 import QtWidgets, QtCore
from matplotlib.figure import Figure
import numpy as np

mlt.use('Qt5Agg')

# 隱藏層的activation function
def relu(x):
    return np.maximum(0, x)

# 類神經網路計算
def MLP(car_state, Whid, Wout):
    front_dist, right_dist, left_dist = car_state
    inp = np.array([[front_dist, right_dist, left_dist]])

    SUMhid = inp @ Whid
    Ahid = relu(SUMhid)

    SUMout = Ahid @ Wout
    # tanh 是輸出層的activation function
    wheel_angle = 40 * np.tanh(SUMout * 1/5)

    return wheel_angle[0, 0]


class Car:
    def __init__(self) -> None:
        self.diameter = 6
        self.angle_min = -90
        self.angle_max = 270
        self.wheel_min = -40
        self.wheel_max = 40
        self.xini_max = 4.5
        self.xini_min = -4.5

        self.reset()

    @property
    def radius(self):
        return self.diameter / 2

    # reset the car to the beginning line
    def reset(self):
        self.angle = 90
        self.wheel_angle = 0

        xini_range = (self.xini_max - self.xini_min - self.diameter)
        left_xpos = self.xini_min + self.diameter // 2
        self.xpos = r.random() * xini_range + left_xpos  # random x pos [-3, 3]
        self.ypos = 0

    def setWheelAngle(self, angle):
        self.wheel_angle = angle if self.wheel_min <= angle <= self.wheel_max else (
            self.wheel_min if angle < self.wheel_min else self.wheel_max)

    def setPosition(self, newPosition: Point2D):
        self.xpos = newPosition.x
        self.ypos = newPosition.y

    # this is the function returning the coordinate on the right, left, front or center points
    def getPosition(self, point='center') -> Point2D:
        if point == 'right':
            right_angle = self.angle - 45
            right_point = Point2D(self.diameter / 2, 0).rotate(right_angle)
            return Point2D(self.xpos, self.ypos) + right_point

        elif point == 'left':
            left_angle = self.angle + 45
            left_point = Point2D(self.diameter / 2, 0).rotate(left_angle)
            return Point2D(self.xpos, self.ypos) + left_point

        elif point == 'front':
            fx = m.cos(self.angle / 180 * m.pi) * self.diameter / 2 + self.xpos
            fy = m.sin(self.angle / 180 * m.pi) * self.diameter / 2 + self.ypos
            return Point2D(fx, fy)
        else:
            return Point2D(self.xpos, self.ypos)

    def setAngle(self, new_angle):
        new_angle %= 360
        if new_angle > self.angle_max:
            new_angle -= self.angle_max - self.angle_min
        self.angle = new_angle

    # set the car state from t to t+1
    def tick(self):
        car_angle = self.angle / 180 * m.pi
        wheel_angle = self.wheel_angle / 180 * m.pi
        new_x = self.xpos + m.cos(car_angle + wheel_angle) + \
                m.sin(wheel_angle) * m.sin(car_angle)

        new_y = self.ypos + m.sin(car_angle + wheel_angle) - \
                m.sin(wheel_angle) * m.cos(car_angle)
        new_angle = (car_angle - m.asin(2 * m.sin(wheel_angle) / self.diameter)) / m.pi * 180

        new_angle %= 360
        if new_angle > self.angle_max:
            new_angle -= self.angle_max - self.angle_min

        self.xpos = new_x
        self.ypos = new_y
        self.setAngle(new_angle)


class Playground:
    def __init__(self):
        # read path lines
        self.path_line_filename = "軌道座標點.txt"
        self._setDefaultLine()
        self.decorate_lines = [
            Line2D(-6, 0, 6, 0),  # start line
            Line2D(0, 0, 0, -3),  # middle line
        ]
        self.complete = False

        self.car = Car()
        self.reset()

    def _setDefaultLine(self):
        # print('use default lines')
        # default lines
        self.destination_line = Line2D(18, 40, 30, 37)

        self.lines = [
            Line2D(-6, -3, 6, -3),
            Line2D(6, -3, 6, 10),
            Line2D(6, 10, 30, 10),
            Line2D(30, 10, 30, 50),
            Line2D(18, 50, 30, 50),
            Line2D(18, 22, 18, 50),
            Line2D(-6, 22, 18, 22),
            Line2D(-6, -3, -6, 22),
        ]

        self.car_init_pos = None
        self.car_init_angle = None

    def _readPathLines(self):
        try:
            with open(self.path_line_filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # get init pos and angle
                pos_angle = [float(v) for v in lines[0].split(',')]
                self.car_init_pos = Point2D(*pos_angle[:2])
                self.car_init_angle = pos_angle[-1]

                # get destination line
                dp1 = Point2D(*[float(v) for v in lines[1].split(',')])
                dp2 = Point2D(*[float(v) for v in lines[2].split(',')])
                self.destination_line = Line2D(dp1, dp2)

                # get wall lines
                self.lines = []
                inip = Point2D(*[float(v) for v in lines[3].split(',')])
                for strp in lines[4:]:
                    p = Point2D(*[float(v) for v in strp.split(',')])
                    line = Line2D(inip, p)
                    inip = p
                    self.lines.append(line)
        except Exception:
            self._setDefaultLine()

    @property
    def n_actions(self):  # action = [0~num_angles-1]
        return (self.car.wheel_max - self.car.wheel_min + 1)

    @property
    def observation_shape(self):
        return (len(self.state),)

    @property
    def state(self):
        front_dist = - 1 if len(self.front_intersects) == 0 else self.car.getPosition(
        ).distToPoint2D(self.front_intersects[0])
        right_dist = - 1 if len(self.right_intersects) == 0 else self.car.getPosition(
        ).distToPoint2D(self.right_intersects[0])
        left_dist = - 1 if len(self.left_intersects) == 0 else self.car.getPosition(
        ).distToPoint2D(self.left_intersects[0])

        return [front_dist, right_dist, left_dist]

    def _checkDoneIntersects(self):
        if self.done:
            return self.done

        cpos = self.car.getPosition('center')  # center point of the car
        cfront_pos = self.car.getPosition('front')
        cright_pos = self.car.getPosition('right')
        cleft_pos = self.car.getPosition('left')
        radius = self.car.radius

        isAtDestination = cpos.isInRect(
            self.destination_line.p1, self.destination_line.p2
        )
        # if we finish the tour
        done = False if not isAtDestination else True
        complete = False if not isAtDestination else True

        front_intersections, find_front_inter = [], True
        right_intersections, find_right_inter = [], True
        left_intersections, find_left_inter = [], True
        for wall in self.lines:  # check every line in play ground
            dToLine = cpos.distToLine2D(wall)
            p1, p2 = wall.p1, wall.p2
            dp1, dp2 = (cpos - p1).length, (cpos - p2).length
            wall_len = wall.length

            # touch conditions
            p1_touch = (dp1 < radius)
            p2_touch = (dp2 < radius)
            body_touch = (
                    dToLine < radius and (dp1 < wall_len and dp2 < wall_len)
            )
            front_touch, front_t, front_u = Line2D(
                cpos, cfront_pos).lineOverlap(wall)
            right_touch, right_t, right_u = Line2D(
                cpos, cright_pos).lineOverlap(wall)
            left_touch, left_t, left_u = Line2D(
                cpos, cleft_pos).lineOverlap(wall)

            if p1_touch or p2_touch or body_touch or front_touch:
                if not done:
                    done = True

            # find all intersections
            if find_front_inter and front_u and 0 <= front_u <= 1:
                front_inter_point = (p2 - p1) * front_u + p1
                if front_t:
                    if front_t > 1:  # select only point in front of the car
                        front_intersections.append(front_inter_point)
                    elif front_touch:  # if overlapped, don't select any point
                        front_intersections = []
                        find_front_inter = False

            if find_right_inter and right_u and 0 <= right_u <= 1:
                right_inter_point = (p2 - p1) * right_u + p1
                if right_t:
                    if right_t > 1:  # select only point in front of the car
                        right_intersections.append(right_inter_point)
                    elif right_touch:  # if overlapped, don't select any point
                        right_intersections = []
                        find_right_inter = False

            if find_left_inter and left_u and 0 <= left_u <= 1:
                left_inter_point = (p2 - p1) * left_u + p1
                if left_t:
                    if left_t > 1:  # select only point in front of the car
                        left_intersections.append(left_inter_point)
                    elif left_touch:  # if overlapped, don't select any point
                        left_intersections = []
                        find_left_inter = False

        self._setIntersections(front_intersections,
                               left_intersections,
                               right_intersections)

        # results
        self.done = done
        self.complete = complete

        return done

    def _setIntersections(self, front_inters, left_inters, right_inters):
        self.front_intersects = sorted(front_inters, key=lambda p: p.distToPoint2D(
            self.car.getPosition('front')))
        self.right_intersects = sorted(right_inters, key=lambda p: p.distToPoint2D(
            self.car.getPosition('right')))
        self.left_intersects = sorted(left_inters, key=lambda p: p.distToPoint2D(
            self.car.getPosition('left')))

    def reset(self):
        self.done = False
        self.complete = False
        self.car.reset()

        if self.car_init_angle and self.car_init_pos:
            self.setCarPosAndAngle(self.car_init_pos, self.car_init_angle)

        self._checkDoneIntersects()
        return self.state

    def setCarPosAndAngle(self, position: Point2D = None, angle=None):
        if position:
            self.car.setPosition(position)
        if angle:
            self.car.setAngle(angle)

        self._checkDoneIntersects()

    def calWheelAngleFromAction(self, action):
        angle = self.car.wheel_min + \
                action * (self.car.wheel_max - self.car.wheel_min) / \
                (self.n_actions - 1)
        return angle

    def step(self, action=None):
        if action:
            # 這裡調整了方向盤的相對位置，但我將其改為了絕對位置
            # 舉例來說：若原先方向盤向左30度，若下一次向右打30度
            # 相對角度的話就變成0度向前，而絕對角度則是變成向右30度
            # angle = self.calWheelAngleFromAction(action=action)
            self.car.setWheelAngle(action)

        if not self.done:
            self.car.tick()
            self._checkDoneIntersects()

            return self.state
        else:
            return self.state

    def run(self, weight):
        Whid = weight[:-1]
        Wout = weight[-1].reshape(-1, 1)
        wheel_angle = MLP(self.state, Whid, Wout)

        self.step(wheel_angle)

# 每個粒子
# (為了避免宣告過於大量的playground使得記憶體不足或是速度過慢，
# 這裡將自走車以外的權重、速度等獨立出來)
class Particle:
    def __init__(self, neuronNumber_hid=50):
        self.neuronNumber_hid = neuronNumber_hid
        self.Whid = np.random.uniform(-10.0, 10.0, size=(3, self.neuronNumber_hid))
        self.Wout = np.random.uniform(-10.0, 10.0, size=(self.neuronNumber_hid, 1))
        self.pre_v = 0
        self.cur_v = 0
        self.rout_history = []
        self.is_successful = False

    # 更新權重
    def update_weight(self, previous_best, neighbor_best, pre_phi=0.5, nei_phi=0.5):
        self.pre_v = self.cur_v

        combined = np.vstack((self.Whid, self.Wout.reshape(1, -1)))

        self.cur_v = self.pre_v + pre_phi * (previous_best - combined) + nei_phi * (neighbor_best - combined)

        self.Whid = self.Whid + self.cur_v[:-1]
        self.Wout = self.Wout + self.cur_v[-1].reshape(-1, 1)

    # 執行car在playground上的計算
    def run_carInPlayground(self, playground: Playground):
        playground.reset()
        self.rout_history.clear()
        self.rout_history.append([playground.car.xpos, playground.car.ypos])
        while not playground.done:
            wheel_angle = MLP(playground.state, self.Whid, self.Wout)
            playground.step(wheel_angle)
            self.rout_history.append([playground.car.xpos, playground.car.ypos])
        self.is_successful = playground.complete

    # 回傳車子的路徑紀錄
    def get_route(self):
        return self.rout_history

    # 回傳這個權重是否可以到達終點
    def is_success(self):
        return self.is_successful

    # 回傳車子的神經元權重
    def get_weight(self):
        return np.vstack((self.Whid, self.Wout.reshape(1, -1)))

# PSO計算
class PSO:
    def __init__(self, init_particleNumber=40):
        self.previous_best_particle_weight = None
        self.previous_best_particle_fitnessVal = float("-inf")
        self.find_successParticle = False

        self.playground = Playground()
        self.particles = [Particle() for i in range(init_particleNumber)]

    # 適應度函數
    def fittness_func(self, car_track: list, is_successful: bool):
        """
        :param car_track: 表示每個粒子的過往軌跡
        :param is_successful: 表示一個粒子是否成功走到終點
        :return: 回傳計算的是適應度值

        score = success score + car last y coordinate - 0.5 * (car total path length)
        """
        self.find_successParticle = True if is_successful else self.find_successParticle

        score = ((200 if is_successful else 0) +  # 成功的路徑出現
                 0.8 * car_track[-1][0] +   # 最終的x值最大
                 car_track[-1][1] -         # 最終的y值最大
                 0.5 * len(car_track))      # 移動路線最短(但考慮到最一開始就失敗的範例，所以將權重減少)

        return score

    def train_model(self):
        while not self.find_successParticle:
            # 先求得每台車子的運行結果
            for particle in self.particles:
                particle.run_carInPlayground(self.playground)

            # 計算歷史最佳和鄰近最佳
            for ind_out, particle in enumerate(self.particles):
                # 更新歷史最佳
                if (self.fittness_func(particle.get_route(), particle.is_success()) >
                        self.previous_best_particle_fitnessVal):
                    self.previous_best_particle_fitnessVal = self.fittness_func(particle.get_route(),
                                                                                particle.is_success())
                    self.previous_best_particle_weight = particle.get_weight()
                    if (particle.is_success()):
                        self.find_successParticle = True

                # 更新鄰居最佳
                neighbor_best_particle_weight = particle.get_weight()
                neighbor_best_particle_fitnessVal = self.fittness_func(particle.get_route(), particle.is_success())
                for ind_in, neighbor in enumerate(self.particles):
                    # 若找到鄰近最佳
                    if (self.fittness_func(neighbor.get_route(), neighbor.is_success()) >
                            neighbor_best_particle_fitnessVal):
                        neighbor_best_particle_weight = neighbor.get_weight()
                        neighbor_best_particle_fitnessVal = self.fittness_func(neighbor.get_route(),
                                                                               neighbor.is_success())
                # 更新粒子的速度和位置
                particle.update_weight(self.previous_best_particle_weight, neighbor_best_particle_weight)

                # print(self.find_successParticle)

    # 取得最佳的粒子
    def get_weight(self):
        return self.previous_best_particle_weight

    def del_particles(self):
        for particle in self.particles:
            del particle
        del self.playground

class AnimationGUI(QtWidgets.QMainWindow):
    '''
    p: playground的建立
    state: 當前狀態
    ani_running: 當下程式是否在執行
    QtCore.QTimer: 控制動畫的執行頻率和狀態
    '''

    def __init__(self, p: Playground, pso: PSO):
        super().__init__()
        self.p = p
        self.state = self.p.reset()
        self.ani_running = False
        self.timer = QtCore.QTimer(self)

        # 訓練模型
        pso.train_model()
        self.weight = pso.get_weight()
        pso.del_particles()

        # 建立主視窗
        self.setWindowTitle("操作介面")
        self.main_widget = QtWidgets.QWidget(self)
        # 將主視窗的中間介面設為main_widget
        self.setCentralWidget(self.main_widget)

        # 建立開始按鈕
        self.start_button = QtWidgets.QPushButton("開始動畫")
        self.start_button.clicked.connect(self.start_animation)

        # 建立繪畫動畫的區域
        # 動畫的底板
        self.figure = Figure(figsize=(5, 5))
        self.canvas = FigureCanvas(self.figure)

        # 建立畫面配置
        # 主要畫面
        layout = QtWidgets.QVBoxLayout(self.main_widget)
        layout.addWidget(self.start_button)
        layout.addWidget(self.canvas)

        # 事前的背景及變數設定
        self.setup_animation()

    def setup_animation(self):
        '''
        ax: 動畫的畫布
        background: 動畫背景
        star_line: 起點
        end_line: 終點
        car_trail: 車子的移動軌跡
        line: 顯示車子當前方向的直線
        text: 當下感測器所偵測到的距離
        '''

        self.ax = self.figure.add_subplot(111)
        self.background = self.p.lines
        self.star_line = self.p.decorate_lines[0]
        self.end_line = self.p.destination_line
        self.car_trail = []
        self.car_radius = self.p.car.radius
        self.line, = self.ax.plot([], [], 'r-')  # 建立圓心到前感測器的直線
        self.text = self.ax.text(15, 0, '', fontsize=10)

        self.draw_background()

    def draw_background(self):
        for line in self.background:
            self.ax.plot([line.p1.x, line.p2.x], [line.p1.y, line.p2.y], "k-")

        # 因為終點是一個矩形，所以需要兩條線
        self.ax.plot([self.end_line.p1.x, self.end_line.p2.x],
                     [self.end_line.p1.y, self.end_line.p1.y], "r-")
        self.ax.plot([self.end_line.p1.x, self.end_line.p2.x],
                     [self.end_line.p2.y, self.end_line.p2.y], "r-")

        # 起點
        self.ax.plot([self.star_line.p1.x, self.star_line.p2.x],
                     [self.star_line.p1.y, self.star_line.p2.y], "b-")

        self.ax.axis('equal')

    # 初始化各項變數後開始動畫
    def start_animation(self):
        if self.ani_running:
            self.timer.stop()

        self.clear_car()
        self.p.reset()
        self.ani_running = True

        # 更新動畫的函數
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(50)  # 每 50 毫秒更新一次

    # 更新動畫的畫面
    def update_animation(self):
        car_pos = self.p.car.getPosition("center")
        self.draw_car(car_pos)
        self.text.set_text(
            f'front censor: {self.p.state[0]:.{3}f}\n'
            f'right censor: {self.p.state[1]:.{3}f}\n'
            f'left censor: {self.p.state[2]:.{3}f}'
        )

        # 如果撞牆的話就算失敗
        # 抵達終點的話就算成功
        if self.p.done:
            if self.p.complete:
                self.show_message("Succeeded!")
            else:
                self.show_message("Failed!")
            self.timer.stop()
            self.ani_running = False

        self.p.run(self.weight)

        # 畫出所有移動畫面
        self.canvas.draw()

    # 畫出車子
    def draw_car(self, car_pos):
        self.car = plt.Circle((car_pos.x, car_pos.y), self.car_radius, color="red", fill=False)
        self.ax.add_patch(self.car)
        self.car_trail.append(self.car)
        front_censor = self.p.car.getPosition("front")
        self.line.set_data([car_pos.x, front_censor.x], [car_pos.y, front_censor.y])

    # 再次開始的時候，清理之前的車子移動軌跡
    def clear_car(self):
        for trace in self.ax.patches:
            trace.remove()
        self.car_trail = []

    # 成功、失敗訊息
    def show_message(self, message):
        msg_box = QtWidgets.QMessageBox()
        msg_box.setText(message)
        msg_box.exec_()

    # 實際顯示整個動畫
    def run(self):
        self.show()


if __name__ == '__main__':
    app = QtWidgets.QApplication([])

    gui = AnimationGUI(Playground(), PSO())

    gui.run()
    # 啟動 PyQt5 應用程式的事件循環。事件循環是一個無限循環,它會接收並處理來自作業系統的事件，
    # 例如鍵盤輸入、滑鼠移動等。只要應用程式沒有被關閉,事件循環就會一直運行。exec_() 方法會阻
    # 塞主線程,直到應用程式退出為止。
    app.exec_()
