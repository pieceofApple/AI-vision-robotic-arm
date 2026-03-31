#include <WiFi.h>
#include <WiFiUdp.h>
#include <math.h>
#include <Arduino.h>

// 机械臂参数定义
#define P 15
#define A1 7.5
#define A2 9.0
#define A3 9.0
#define A4 3.8//实际为3.5,舵机死区导致下垂

#ifndef M_PI
#define M_PI 3.141592653
#endif

#define MAX_LEN (A2 + A3)
#define MAX_HIGH (A1 + A2 + A3)

#define DEG_TO_RAD(deg) ((deg) * M_PI / 180.0)
#define RAD_TO_DEG(rad) ((rad) * 180.0 / M_PI)

// 新增状态变量
enum State {
    IDLE,         // 空闲状态
    MOVING,       // 移动中
    GRABBING,     // 抓取中
    PLACING       // 放置中
};
State currentState = IDLE;
bool actionComplete = true; // 动作完成标志

// 要连接的热点配置（请替换为实际 WiFi）
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// UDP 相关设置
WiFiUDP udp;
const int localPort = 8080;  // 本地监听端口
const char* remoteIP = "192.168.1.100";  // 远程 Maixcam 等 IP，请按实际修改
const int remotePort = 8080;  // 远程端口
char incomingPacket[255];

// 计算三角函数
double cos_deg(double degree) {
    return cos(DEG_TO_RAD(degree));
}

double sin_deg(double degree) {
    return sin(DEG_TO_RAD(degree));
}

double atan2_deg(double y, double x) {
    return RAD_TO_DEG(atan2(y, x));
}

// 关节角度转换
double j_degree_convert(int joint, double j_or_deg) {
    if (joint == 1) {
        return j_or_deg;
    } else if (joint == 2 || joint == 3 || joint == 4) {
        return 90.0 - j_or_deg;
    } else {
        return -1; // 错误值
    }
}

// 验证角度是否有效
bool valid_degree(double degree) {
    return (degree >= 0.0 && degree <= 180.0);
}

// 检查是否超出范围
bool out_of_range(double length, double height) {
    return (height > MAX_HIGH || length > MAX_LEN);
}

// 计算 j1
void calculate_j1(double x, double y, double z, double *j1, double *length, double *height) {
    *length = sqrt(pow(y + P, 2) + pow(x, 2));
    *j1 = (*length == 0) ? 0 : atan2_deg(y + P, x);
    *height = z;
}

// 计算 j3
double calculate_j3(double L, double H) {
    double cos3 = (pow(L, 2) + pow(H, 2) - pow(A2, 2) - pow(A3, 2)) / (2 * A2 * A3);
    if (cos3 > 1.0 || cos3 < -1.0) return -1; // 无解
    double sin3 = sqrt(1 - pow(cos3, 2));
    return atan2_deg(sin3, cos3);
}

// 计算 j2
double calculate_j2(double L, double H, double j3) {
    double K1 = A2 + A3 * cos_deg(j3);
    double K2 = A3 * sin_deg(j3);
    double w = atan2_deg(K2, K1);
    return atan2_deg(L, H) - w;
}

// 逆运动学计算
bool backward_kinematics(double x, double y, double z, double alpha, double *deg1, double *deg2, double *deg3, double *deg4) {
    double j1, j2, j3, j4;
    double length, height;
    calculate_j1(x, y, z + A4, &j1, &length, &height);

    if (out_of_range(length, height)) return false;

    double L = length;
    double H = height - A1;
    j3 = calculate_j3(L, H);
    j2 = calculate_j2(L, H, j3);
    j4 = alpha - j2 - j3;

    *deg1 = round(j_degree_convert(1, j1) * 100) / 100.0;
    *deg2 = round(j_degree_convert(2, j2) * 100) / 100.0;
    *deg3 = round(j_degree_convert(3, j3) * 100) / 100.0;
    *deg4 = round(j_degree_convert(4, j4) * 100) / 100.0;

    return true;
}

// 正运动学计算
bool forward_kinematics(double deg1, double deg2, double deg3, double deg4, double *x, double *y, double *z) {
    if (!valid_degree(deg1) || !valid_degree(deg2) || !valid_degree(deg3) || !valid_degree(deg4)) {
        return false;
    }

    double j1 = j_degree_convert(1, deg1);
    double j2 = j_degree_convert(2, deg2);
    double j3 = j_degree_convert(3, deg3);
    double j4 = j_degree_convert(4, deg4);

    double length = A2 * sin_deg(j2) + A3 * sin_deg(j2 + j3) + A4 * sin_deg(j2 + j3 + j4);
    double height = A1 + A2 * cos_deg(j2) + A3 * cos_deg(j2 + j3) + A4 * cos_deg(j2 + j3 + j4);

    *z = round(height * 100) / 100.0;
    *x = round(length * cos_deg(j1) * 100) / 100.0;
    *y = round((length * sin_deg(j1) - P) * 100) / 100.0;

    return (*y >= 0 && *z >= 0);
}

// 定义要使用的引脚，用于控制5个舵机
const int pwmPins[] = {33, 25, 26, 27, 12};
// 定义状态指示灯引脚
const int statusLedPin = 2;
// 定义PWM频率，舵机常用50Hz
const int freq = 50;
// 定义PWM分辨率
const int resolution = 12;
// 占空比数值范围，12位分辨率下为0 - 4095
const int dutyMax = (1 << resolution) - 1;
// 舵机最小占空比对应的数值，在50Hz频率下，0.4ms对应的值
const int servoMinDuty = (int)(0.4 / (1000.0 / freq) * dutyMax);
// 舵机最大占空比对应的数值，在50Hz频率下，2.4ms对应的值
const int servoMaxDuty = (int)(2.4 / (1000.0 / freq) * dutyMax);
// 定义各个舵机的初始角度
const int initialAngles[] = {100, 90-60, 92+60, 60, 20}; //直角90°，实际相对于Z轴{N,0,55,90,N}
//const int initialAngles[] = {100, 90, 39, 89, 20};//平行z轴，相当于初始角度{90,0,0,0,N}
// 闪烁间隔时间（毫秒）
const int blinkInterval = 500;
// 上次闪烁时间
unsigned long previousMillis = 0;
// 指示灯状态
bool ledState = LOW;

// 控制舵机角度
void setServoAngles(float angles[4]) {
    for (int i = 0; i < 4; i++) {
        if (valid_degree(angles[i])) {
            int duty = map((int)angles[i], 0, 180, servoMinDuty, servoMaxDuty);
            ledcWrite(pwmPins[i], duty);
            // Serial.print("Pin ");
            // Serial.print(pwmPins[i]);
            // Serial.print(" angle set to: ");
            // Serial.println(angles[i]);
        } else {
            // Serial.print("Angle at index ");
            // Serial.print(i);
            // Serial.print(" is invalid. Angle value: ");
            // Serial.println(angles[i]);
        }
    }
}

// 控制单个舵机角度
void setSingleServoAngle(int pin, int angle) {
    if (valid_degree(angle)) {
        int duty = map(angle, 0, 180, servoMinDuty, servoMaxDuty);
        ledcWrite(pin, duty);
        // Serial.print("Pin ");
        // Serial.print(pin);
        // Serial.print(" angle set to: ");
        // Serial.println(angle);
    } else {
        // Serial.print("Invalid angle for pin ");
        // Serial.print(pin);
        // Serial.print(": ");
        // Serial.println(angle);
    }
}

// 执行运动到指定位置
bool moveToPosition(double x, double y, double z) {
    double deg1, deg2, deg3, deg4;
    if (backward_kinematics(x, y, z, 180.0, &deg1, &deg2, &deg3, &deg4)) {
        float angles[4] = {100 - (90 - deg1), 180 - deg2, 31 + (90 - deg3), 89 - (90 - deg4)};
        setServoAngles(angles);
        // 添加延时，确保舵机有足够时间转动
        delay(1000); 
        return true;
    } else {
        Serial.println("逆运动学计算失败，请检查输入的坐标值。");
        return false;
    }
}

// 修改后的 UDP 请求逻辑
void triggerUDPRequest() {
    if (actionComplete && currentState == IDLE) {
        udp.beginPacket(remoteIP, remotePort);
        udp.print("请求数据");
        udp.endPacket();
        Serial.println("已发送请求数据");
        actionComplete = false; // 标记请求已发送，禁止重复触发
    }
}

// 抓取动作（新增完成回调）
void grabAction(double x, double y, double z) {
    currentState = GRABBING;
    // 松开爪子
    setSingleServoAngle(12, 90);
    delay(500);

    // 抬升至 z = 5
    if (moveToPosition(x, y, 5)) {
        // 移动到抓取位置
        if (moveToPosition(x, y, z)) {
            // 抓紧爪子
            setSingleServoAngle(12, 20);
            delay(500);

            // 抬升至安全高度
            if (moveToPosition(x, y, 5)) {

            }
        }
    }
}

// 放置动作（新增完成回调）
void placeAction() {
    currentState = PLACING;
    double x = 15;
    double y = -15;
    double z = 4;

    // 移动到放置位置上方
    if (moveToPosition(x, y, 5)) {
        // 移动到放置位置
        if (moveToPosition(x, y, z)) {
            // 松开爪子
            setSingleServoAngle(12, 90);
            delay(500);

            // 抬升至安全高度
            if (moveToPosition(x, y, 5)) {
                // currentState = IDLE;
                // actionComplete = true; // 放置完成，允许触发下一次请求
                // triggerUDPRequest(); // 自动触发下一次数据请求（可选）
            }
        }
    }
}

// 机械臂回归初始位置
void resetArm() {
    for (int i = 0; i < 5; i++) {
        int initialDuty = map(initialAngles[i], 0, 180, servoMinDuty, servoMaxDuty);
        ledcWrite(pwmPins[i], initialDuty);
        Serial.print("Pin ");
        Serial.print(pwmPins[i]);
        Serial.print(" reset to initial angle: ");
        Serial.println(initialAngles[i]);
        delay(500); // 每个舵机动作之间添加延时
    }
    actionComplete = true; // 请求数据
    currentState = IDLE;
}

// 处理 UDP 消息
void handleUDPMessage() {
    int packetSize = udp.parsePacket();
    if (packetSize) {
        Serial.print("收到来自 ");
        Serial.print(udp.remoteIP());
        Serial.print(":");
        Serial.print(udp.remotePort());
        Serial.print(" 的数据包，大小为 ");
        Serial.println(packetSize);

        // 读取数据包
        int len = udp.read(incomingPacket, 255);
        if (len > 0) {
            incomingPacket[len] = '\0';
        }
        Serial.print("收到消息: ");
        Serial.println(incomingPacket);

        String input = String(incomingPacket);
        input.trim();

        if (input == "rst") {
            resetArm();
        } else if (input == "stop") { // 添加终止信号处理
            // 停止请求数据的逻辑
            actionComplete = false;
            currentState = IDLE;
        } else {
            // 检查输入字符串是否包含三个逗号
            int commaCount = 0;
            for (int i = 0; i < input.length(); i++) {
                if (input[i] == ',') {
                    commaCount++;
                }
            }

            if (commaCount == 3) {
                // 查找逗号位置
                int comma1 = input.indexOf(',');
                int comma2 = input.indexOf(',', comma1 + 1);
                int comma3 = input.indexOf(',', comma2 + 1);

                // 提取四个值，使用toFloat()处理小数
                float x = input.substring(0, comma1).toFloat();
                float y = input.substring(comma1 + 1, comma2).toFloat();
                float z = input.substring(comma2 + 1, comma3).toFloat();
                int flag = input.substring(comma3 + 1).toInt();

                Serial.print("Received x: ");
                Serial.print(x);
                Serial.print(", y: ");
                Serial.print(y);
                Serial.print(", z: ");
                Serial.print(z);
                Serial.print(", flag: ");
                Serial.println(flag);

                if (flag == 1) {
                    // 执行抓取动作
                    grabAction(x, y, z);
                    // 执行放置动作
                    placeAction();
                    //防止遮挡摄像头视线
                    resetArm();
                } else if (flag == 0) { // 新增逻辑，当flag为0时移动到指定位置
                    moveToPosition(x, y, z);
                } else {
                    Serial.println("Flag value is invalid, skipping actions.");
                }
            } else {
                Serial.println("输入格式错误，请使用 'x,y,z,flag' 格式或输入 'rst' 复位。");
            }
        }
    }
}

void setup() {
    // 初始化串口通信
    Serial.begin(115200);
    // 设置状态指示灯引脚为输出模式
    pinMode(statusLedPin, OUTPUT);
    // 初始化每个引脚的PWM
    for (int i = 0; i < 5; i++) {
        // 设置PWM引脚的频率和分辨率
        ledcAttach(pwmPins[i], freq, resolution);
        // 将初始角度转换为对应的占空比数值
        int initialDuty = map(initialAngles[i], 0, 180, servoMinDuty, servoMaxDuty);
        // 设置舵机到初始角度
        ledcWrite(pwmPins[i], initialDuty);
    }

    // 连接到热点
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(1000);
        Serial.println("正在连接到热点...");
    }
    Serial.println("已连接到热点");
    Serial.print("本地 IP 地址: ");
    Serial.println(WiFi.localIP());

    // 启动 UDP
    if (udp.begin(localPort) == 1) {
        Serial.println("UDP 客户端已启动，等待消息...");
    } else {
        Serial.println("UDP 客户端启动失败");
    }
}

void loop() {
    // 获取当前时间
    unsigned long currentMillis = millis();
    // 检查是否到了闪烁时间
    if (currentMillis - previousMillis >= blinkInterval) {
        // 记录上次闪烁时间
        previousMillis = currentMillis;
        // 切换指示灯状态
        ledState = !ledState;
        // 设置指示灯状态
        digitalWrite(statusLedPin, ledState);
    }

    // 优先处理串口输入
    if (Serial.available() > 0) {
        // 用于存储接收到的整行数据
        String input = Serial.readStringUntil('\n');
        // 清理输入字符串，去除首尾空格和换行符
        input.trim();

        if (input == "rst") {
            resetArm();
            // 通过 UDP 发送 rst 指令
            udp.beginPacket(remoteIP, remotePort);
            udp.print("rst");
            udp.endPacket();
            Serial.println("已通过 UDP 发送 rst 指令");
            actionComplete = false; // 停止请求数据
            currentState = IDLE;
            // 清空串口缓冲区，避免残留数据影响后续解析
            while (Serial.available() > 0) {
                Serial.read();
            }
            return;
        } else if (input == "send") {
            actionComplete = true; // 重新开始请求数据
            triggerUDPRequest();
            // 清空串口缓冲区，避免残留数据影响后续解析
            while (Serial.available() > 0) {
                Serial.read();
            }
            return;
        } else {
            // 检查输入字符串是否包含三个逗号
            int commaCount = 0;
            for (int i = 0; i < input.length(); i++) {
                if (input[i] == ',') {
                    commaCount++;
                }
            }

            if (commaCount == 3) {
                // 查找逗号位置
                int comma1 = input.indexOf(',');
                int comma2 = input.indexOf(',', comma1 + 1);
                int comma3 = input.indexOf(',', comma2 + 1);

                // 提取四个值，使用toFloat()处理小数
                float x = input.substring(0, comma1).toFloat();
                float y = input.substring(comma1 + 1, comma2).toFloat();
                float z = input.substring(comma2 + 1, comma3).toFloat();
                int flag = input.substring(comma3 + 1).toInt();

                Serial.print("Received x: ");
                Serial.print(x);
                Serial.print(", y: ");
                Serial.print(y);
                Serial.print(", z: ");
                Serial.print(z);
                Serial.print(", flag: ");
                Serial.println(flag);

                if (flag == 1) {
                    // 执行抓取动作
                    grabAction(x, y, z);
                    // 执行放置动作
                    placeAction();
                    //防止遮挡摄像头视线
                    resetArm();
                } else if (flag == 0) { // 新增逻辑，当flag为0时移动到指定位置
                    moveToPosition(x, y, z);
                } else {
                    Serial.println("Flag value is invalid, skipping actions.");
                }
            } else {
                Serial.println("输入格式错误，请使用 'x,y,z,flag' 格式或输入 'rst' 复位。");
            }
            // 清空串口缓冲区，避免残留数据影响后续解析
            while (Serial.available() > 0) {
                Serial.read();
            }
        }
    }

    handleUDPMessage(); // 处理 UDP 数据

    // 仅在空闲状态时，根据动作完成标志触发请求
    if (currentState == IDLE && actionComplete) {
        // 首次启动时主动触发一次请求（可选）
        // static bool firstTrigger = true;//该方法只有第一次才会触发，后续firstTrigger=False
        // if (firstTrigger) {
        //     triggerUDPRequest();
        //     firstTrigger = false;
        // }
        triggerUDPRequest();
    }

    delay(10);
}
    