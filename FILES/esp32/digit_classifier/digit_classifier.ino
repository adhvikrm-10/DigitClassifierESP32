#include <Arduino.h>
#include "model_weights.h"

static const int PIN_LED_BUILTIN = 2;

static const int DIGIT_PINS[OUTPUT_SIZE] = {
  13, 12, 14, 27, 26, 25, 33, 32, 4, 5
};

static const int PIN_BTN_CONFIRM   = 18;
static const int PIN_BTN_INCREMENT = 19;

static const uint8_t CMD_INFER   = 0xFF;
static const uint8_t CMD_CORRECT = 0xFE;
static const float   LR          = 0.05f;

static float   input_buf [INPUT_SIZE ];
static float   hidden_buf[HIDDEN_SIZE];
static float   output_buf[OUTPUT_SIZE];

static float   w2_ram[HIDDEN_SIZE * OUTPUT_SIZE];
static float   b2_ram[OUTPUT_SIZE];

static uint8_t last_image[INPUT_SIZE];
static int     last_prediction = -1;

static uint32_t btn_confirm_last   = 0;
static uint32_t btn_increment_last = 0;
static int      display_digit      = 0;

static int  run_inference(const uint8_t *img);
static void apply_correction(int true_label);
static void light_digit(int digit);
static void blink_led(int n);

void setup() {
  Serial.begin(115200);
  while (!Serial) { delay(10); }

  pinMode(PIN_LED_BUILTIN, OUTPUT);

  for (int i = 0; i < OUTPUT_SIZE; i++) {
    pinMode(DIGIT_PINS[i], OUTPUT);
    digitalWrite(DIGIT_PINS[i], LOW);
  }

  pinMode(PIN_BTN_CONFIRM,   INPUT_PULLUP);
  pinMode(PIN_BTN_INCREMENT, INPUT_PULLUP);

  for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++) {
    w2_ram[i] = pgm_read_float_near(&W2[i]);
  }

  for (int i = 0; i < OUTPUT_SIZE; i++) {
    b2_ram[i] = pgm_read_float_near(&b2[i]);
  }

  blink_led(3);
  Serial.println("ESP32 Perceptron Classifier ready.");
}

void loop() {
  if (Serial.available()) {
    uint8_t cmd = (uint8_t)Serial.read();

    if (cmd == CMD_INFER) {
      uint32_t t0 = millis();

      while (Serial.available() < INPUT_SIZE) {
        if (millis() - t0 > 3000) {
          Serial.println("TIMEOUT");
          return;
        }
      }

      Serial.readBytes((char *)last_image, INPUT_SIZE);

      int pred = run_inference(last_image);
      last_prediction = pred;

      Serial.println(pred);
      light_digit(pred);
      blink_led(1);

    } else if (cmd == CMD_CORRECT) {
      uint32_t t0 = millis();

      while (Serial.available() < 1) {
        if (millis() - t0 > 2000) {
          Serial.println("TIMEOUT");
          return;
        }
      }

      uint8_t true_label = (uint8_t)Serial.read();

      if (true_label > 9) {
        Serial.println("ERR:bad_label");
        return;
      }

      apply_correction((int)true_label);

      Serial.print("OK:");
      Serial.println(true_label);
    }
  }

  if (digitalRead(PIN_BTN_CONFIRM) == LOW) {
    uint32_t now = millis();

    if (now - btn_confirm_last > 300) {
      btn_confirm_last = now;

      if (last_prediction >= 0) {
        Serial.println(last_prediction);
        light_digit(last_prediction);
        blink_led(1);
      }
    }
  }

  if (digitalRead(PIN_BTN_INCREMENT) == LOW) {
    uint32_t now = millis();

    if (now - btn_increment_last > 300) {
      btn_increment_last = now;
      display_digit = (display_digit + 1) % OUTPUT_SIZE;
      light_digit(display_digit);
    }
  }
}

static int run_inference(const uint8_t *img) {
  for (int i = 0; i < INPUT_SIZE; i++) {
    input_buf[i] = img[i] / 255.0f;
  }

  for (int j = 0; j < HIDDEN_SIZE; j++) {
    float sum = pgm_read_float_near(&b1[j]);

    for (int i = 0; i < INPUT_SIZE; i++) {
      sum += input_buf[i] * pgm_read_float_near(&W1[i * HIDDEN_SIZE + j]);
    }

    hidden_buf[j] = (sum > 0.0f) ? sum : 0.0f;
  }

  int   best_idx   = 0;
  float best_score = -1e9f;

  for (int k = 0; k < OUTPUT_SIZE; k++) {
    float sum = b2_ram[k];

    for (int j = 0; j < HIDDEN_SIZE; j++) {
      sum += hidden_buf[j] * w2_ram[j * OUTPUT_SIZE + k];
    }

    output_buf[k] = sum;

    if (sum > best_score) {
      best_score = sum;
      best_idx   = k;
    }
  }

  return best_idx;
}

static void apply_correction(int true_label) {
  if (last_prediction < 0) return;

  int pred = run_inference(last_image);

  if (pred == true_label) return;

  for (int j = 0; j < HIDDEN_SIZE; j++) {
    w2_ram[j * OUTPUT_SIZE + pred]       -= LR * hidden_buf[j];
    w2_ram[j * OUTPUT_SIZE + true_label] += LR * hidden_buf[j];
  }

  b2_ram[pred]       -= LR;
  b2_ram[true_label] += LR;
}

static void light_digit(int digit) {
  for (int i = 0; i < OUTPUT_SIZE; i++) {
    digitalWrite(DIGIT_PINS[i], (i == digit) ? HIGH : LOW);
  }
}

static void blink_led(int n) {
  for (int i = 0; i < n; i++) {
    digitalWrite(PIN_LED_BUILTIN, HIGH);
    delay(80);
    digitalWrite(PIN_LED_BUILTIN, LOW);
    delay(80);
  }
}
