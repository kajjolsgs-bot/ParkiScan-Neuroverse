/*
  ParkiScan IMU node — ESP32 + MPU-6050, newline-delimited JSON over USB serial.
  ============================================================================
  THIS COPY IS CONFIGURED FOR:  WRIST     (wrist and ankle sketches differ only in SENSOR_PLACEMENT)

  Board (Arduino IDE): Tools → Board → esp32 → "ESP32 Dev Module"   (classic ESP32 / ESP32-WROOM-32, CP2102 USB)
  Upload speed 921600 (lower to 115200 if upload fails). Serial Monitor: 921600 baud, "Newline".
  Libraries: none beyond the ESP32 core (uses Wire.h only).

  Wiring (GY-521 MPU-6050 breakout → ESP32 DevKit):
      VCC → 3V3      GND → GND      SDA → GPIO21      SCL → GPIO22      AD0 → GND (address 0x68; AD0 → 3V3 gives 0x69)
      INT, XDA, XCL: not connected.

  Output — one JSON object per line, two kinds:
    data   {"type":"imu","v":2,"dev":"esp32-wrist-a1b2c3","placement":"wrist","seq":42,"t_ms":12345,"t_us":12345678,
            "ax":0.0123,"ay":-0.9812,"az":0.1050,"gx":1.252,"gy":-0.397,"gz":0.305,"sim":false}
    status {"type":"status","v":2,"dev":"...","placement":"wrist","state":"ready","msg":"...", ...}
  Units: acceleration in g (±4 g range), angular rate in deg/s (±500 deg/s range), DLPF 44 Hz, 100 Hz.
  Values are the sensor's raw scaled readings — no bias is subtracted on the device, nothing is smoothed or invented.
  If the sensor is missing, stops answering, or returns frozen values, NO data lines are sent — only status lines.

  Serial commands (send one character): 'p' pause/resume · 'i' info · 'c' still-check (reports gyro bias; keep still 2 s)
                                        'r' re-initialise the sensor
  Research data logger — not a medical device.
*/
#include <Arduino.h>
#include <Wire.h>

// ----------------------------------------------------------------------------- configuration
#define SENSOR_PLACEMENT "wrist"     // "wrist" or "ankle" — identifies this stream to Python and Unity
#define SAMPLE_HZ        100
#define SERIAL_BAUD      921600
#define SDA_PIN          21
#define SCL_PIN          22
#define FW_VERSION       "2.0.0"

// ----------------------------------------------------------------------------- MPU-6050 registers
static const uint8_t REG_SMPLRT_DIV = 0x19, REG_CONFIG = 0x1A, REG_GYRO_CONFIG = 0x1B, REG_ACCEL_CONFIG = 0x1C,
                     REG_ACCEL_XOUT_H = 0x3B, REG_PWR_MGMT_1 = 0x6B, REG_WHO_AM_I = 0x75;
static const float ACC_LSB_PER_G = 8192.0f;    // ±4 g
static const float GYR_LSB_PER_DPS = 65.5f;    // ±500 deg/s

uint8_t  mpuAddr = 0;            // 0x68 / 0x69 once found, 0 = not found
uint8_t  whoAmI = 0;
bool     sensorOk = false, paused = false;
uint32_t seq = 0, sentSinceBeat = 0, readErrors = 0, readErrorsTotal = 0, stuckCount = 0;
uint32_t lastBeatMs = 0, lastRetryMs = 0;
int16_t  lastRaw[6] = {0, 0, 0, 0, 0, 0};
char     devId[40];

// ----------------------------------------------------------------------------- helpers
void status(const char* state, const char* msg) {
  Serial.printf("{\"type\":\"status\",\"v\":2,\"dev\":\"%s\",\"placement\":\"%s\",\"state\":\"%s\",\"msg\":\"%s\","
                "\"fw\":\"%s\",\"addr\":\"0x%02X\",\"who_am_i\":\"0x%02X\",\"t_ms\":%lu}\n",
                devId, SENSOR_PLACEMENT, state, msg, FW_VERSION, mpuAddr, whoAmI, (unsigned long)millis());
}

bool writeReg(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(mpuAddr); Wire.write(reg); Wire.write(val);
  return Wire.endTransmission() == 0;
}

bool readRegs(uint8_t reg, uint8_t* buf, uint8_t n) {
  Wire.beginTransmission(mpuAddr); Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((uint16_t)mpuAddr, (uint8_t)n) != n) return false;
  for (uint8_t i = 0; i < n; i++) buf[i] = Wire.read();
  return true;
}

bool probe(uint8_t addr) {
  Wire.beginTransmission(addr);
  return Wire.endTransmission() == 0;
}

/** Find and configure the sensor. Returns false (with a status line explaining why) if it cannot be used. */
bool initSensor() {
  sensorOk = false;
  mpuAddr = probe(0x68) ? 0x68 : (probe(0x69) ? 0x69 : 0);
  if (!mpuAddr) {
    whoAmI = 0;
    status("sensor_error", "No device answers at I2C 0x68 or 0x69. Check VCC->3V3, GND->GND, SDA->GPIO21, SCL->GPIO22.");
    return false;
  }
  if (!readRegs(REG_WHO_AM_I, &whoAmI, 1)) { status("sensor_error", "Device found but WHO_AM_I read failed (loose wire?)"); return false; }
  if (whoAmI == 0x00 || whoAmI == 0xFF) { status("sensor_error", "WHO_AM_I invalid (0x00/0xFF): I2C bus problem"); return false; }
  bool ok = writeReg(REG_PWR_MGMT_1, 0x80);            // device reset
  delay(100);
  ok &= writeReg(REG_PWR_MGMT_1, 0x01);                // wake, PLL with X-gyro reference
  delay(20);
  ok &= writeReg(REG_CONFIG, 0x03);                    // DLPF ~44 Hz
  ok &= writeReg(REG_SMPLRT_DIV, 0x01);                // 1 kHz / (1+1) = 500 Hz internal: every 100 Hz read gets a fresh sample
  ok &= writeReg(REG_GYRO_CONFIG, 0x08);               // ±500 deg/s
  ok &= writeReg(REG_ACCEL_CONFIG, 0x08);              // ±4 g
  if (!ok) { status("sensor_error", "Configuration writes failed"); return false; }
  delay(50);
  sensorOk = true; readErrors = 0; stuckCount = 0;
  if (whoAmI == 0x68) status("ready", "MPU-6050 detected and configured: 100 Hz, +-4 g, +-500 deg/s, DLPF 44 Hz");
  else status("ready_warning", "Sensor answers with WHO_AM_I other than 0x68 (MPU-6050 clone/MPU-6500 family?). Register map compatible; verify readings.");
  return true;
}

bool readSample(int16_t* raw) {
  uint8_t b[14];
  if (!readRegs(REG_ACCEL_XOUT_H, b, 14)) return false;
  raw[0] = (int16_t)((b[0] << 8) | b[1]);  raw[1] = (int16_t)((b[2] << 8) | b[3]);  raw[2] = (int16_t)((b[4] << 8) | b[5]);
  // b[6..7] = temperature (not used)
  raw[3] = (int16_t)((b[8] << 8) | b[9]);  raw[4] = (int16_t)((b[10] << 8) | b[11]); raw[5] = (int16_t)((b[12] << 8) | b[13]);
  return true;
}

/** Still-check: averages the gyro for 2 s and reports the bias and noise. Does not alter the streamed values. */
void stillCheck() {
  if (!sensorOk) { status("sensor_error", "still-check skipped: sensor not ready"); return; }
  double s[3] = {0, 0, 0}, ss[3] = {0, 0, 0}; int n = 0; int16_t r[6];
  for (int i = 0; i < 200; i++) {
    if (readSample(r)) { for (int k = 0; k < 3; k++) { double g = r[3 + k] / GYR_LSB_PER_DPS; s[k] += g; ss[k] += g * g; } n++; }
    delay(10);
  }
  if (n < 50) { status("sensor_error", "still-check failed: too few readings"); return; }
  double sd = 0; for (int k = 0; k < 3; k++) sd += sqrt(fmax(0.0, ss[k] / n - (s[k] / n) * (s[k] / n)));
  Serial.printf("{\"type\":\"status\",\"v\":2,\"dev\":\"%s\",\"placement\":\"%s\",\"state\":\"still_check\",\"msg\":\"gyro bias (deg/s) measured while still\","
                "\"gyro_bias\":[%.3f,%.3f,%.3f],\"gyro_noise_sd\":%.3f,\"still\":%s,\"n\":%d,\"t_ms\":%lu}\n",
                devId, SENSOR_PLACEMENT, s[0] / n, s[1] / n, s[2] / n, sd / 3, (sd / 3 < 0.5) ? "true" : "false", n, (unsigned long)millis());
}

// ----------------------------------------------------------------------------- setup / loop
void setup() {
  Serial.setTxBufferSize(4096);           // a full line is ~230 bytes; never block the sampling loop
  Serial.begin(SERIAL_BAUD);
  delay(400);
  uint64_t mac = ESP.getEfuseMac();
  snprintf(devId, sizeof(devId), "esp32-%s-%06x", SENSOR_PLACEMENT, (unsigned)((mac >> 24) & 0xFFFFFF));
  Wire.begin(SDA_PIN, SCL_PIN, 400000);
  Wire.setTimeOut(20);
  status("booting", "ParkiScan IMU firmware starting");
  initSensor();
  lastBeatMs = millis();
}

void loop() {
  static uint32_t next = micros();
  const uint32_t period = 1000000UL / SAMPLE_HZ;

  while (Serial.available()) {
    char c = Serial.read();
    if (c == 'p') { paused = !paused; status(paused ? "paused" : "streaming", paused ? "streaming paused by command" : "streaming resumed"); }
    else if (c == 'i') status(sensorOk ? "info" : "sensor_error", sensorOk ? "sensor ok" : "sensor not available");
    else if (c == 'c') stillCheck();
    else if (c == 'r') initSensor();
  }

  uint32_t nowMs = millis();
  // heartbeat once per second: lets the receiver tell "board alive, sensor missing" from "board unplugged"
  if (nowMs - lastBeatMs >= 1000) {
    Serial.printf("{\"type\":\"status\",\"v\":2,\"dev\":\"%s\",\"placement\":\"%s\",\"state\":\"%s\",\"msg\":\"heartbeat\","
                  "\"rate_hz\":%.1f,\"read_errors\":%lu,\"t_ms\":%lu}\n",
                  devId, SENSOR_PLACEMENT, paused ? "paused" : (sensorOk ? "heartbeat" : "sensor_error"),
                  sentSinceBeat * 1000.0f / (nowMs - lastBeatMs), (unsigned long)readErrorsTotal, (unsigned long)nowMs);
    sentSinceBeat = 0; lastBeatMs = nowMs;
  }

  if (!sensorOk) {                           // keep trying to (re)connect every 2 s; never send data without a sensor
    if (nowMs - lastRetryMs >= 2000) { lastRetryMs = nowMs; if (initSensor()) next = micros(); }
    return;
  }

  if ((int32_t)(micros() - next) < 0) return;
  next += period;
  if ((int32_t)(micros() - next) > (int32_t)(period * 10)) next = micros() + period;   // fell far behind: resync, gap shows in seq
  if (paused) return;

  int16_t r[6];
  uint32_t tUs = micros(), tMs = millis();
  uint32_t thisSeq = seq++;                  // seq advances even on failure, so lost samples are visible downstream
  if (!readSample(r)) {
    readErrors++; readErrorsTotal++;
    if (readErrors >= 20) { status("sensor_lost", "20 consecutive I2C read failures: sensor disconnected?"); sensorOk = false; lastRetryMs = nowMs; }
    return;
  }
  readErrors = 0;
  // a real MPU-6050 always has a few LSB of noise: identical raw values for 1 s means the bus/sensor is frozen
  bool same = true; for (int k = 0; k < 6; k++) if (r[k] != lastRaw[k]) { same = false; break; }
  memcpy(lastRaw, r, sizeof(lastRaw));
  stuckCount = same ? stuckCount + 1 : 0;
  if (stuckCount >= SAMPLE_HZ) { status("sensor_stuck", "readings frozen for 1 s (real sensors always show noise): re-initialising sensor"); initSensor(); return; }
  if (stuckCount > 0) return;                // do not emit a value identical to the previous one until it changes or recovers

  Serial.printf("{\"type\":\"imu\",\"v\":2,\"dev\":\"%s\",\"placement\":\"%s\",\"seq\":%lu,\"t_ms\":%lu,\"t_us\":%lu,"
                "\"ax\":%.4f,\"ay\":%.4f,\"az\":%.4f,\"gx\":%.3f,\"gy\":%.3f,\"gz\":%.3f,\"sim\":false}\n",
                devId, SENSOR_PLACEMENT, (unsigned long)thisSeq, (unsigned long)tMs, (unsigned long)tUs,
                r[0] / ACC_LSB_PER_G, r[1] / ACC_LSB_PER_G, r[2] / ACC_LSB_PER_G,
                r[3] / GYR_LSB_PER_DPS, r[4] / GYR_LSB_PER_DPS, r[5] / GYR_LSB_PER_DPS);
  sentSinceBeat++;
}
