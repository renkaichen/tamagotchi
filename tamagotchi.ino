  #include <Preferences.h>
  #include "driver/rtc_io.h"
  #include <Adafruit_GFX.h>
  #include <Adafruit_ST7789.h>
  #include <SPI.h>
  #include <LittleFS.h>
  #include "llama2_runner.hpp"
  #include "freertos/FreeRTOS.h"
  #include "freertos/task.h"
  #include "freertos/queue.h"


  Llama2Engine engine;

  struct PromptMessage {
    char text[192];
  };

  QueueHandle_t promptQueue = nullptr;

  #define BABY 0
  #define KID 1
  #define ADULT 2

  #ifndef BOARD_LED_PIN
  #define BOARD_LED_PIN 38
  #endif

  #ifndef DECAY_INTERVAL_MS
  #define DECAY_INTERVAL_MS 2500
  #endif

  #ifndef EXP_AMOUNT
  #define EXP_AMOUNT 5
  #endif

  #ifndef PET_NAME
  #define PET_NAME LeBron
  #endif

  #ifndef START_HAPPINESS
  #define START_HAPPINESS 100
  #endif

  #ifndef START_HUNGER
  #define START_HUNGER 0
  #endif

  #ifndef START_HEALTH
  #define START_HEALTH 100
  #endif

  // gpt code to parse string
  #define STRINGIFY_INNER(value) #value
  #define STRINGIFY(value) STRINGIFY_INNER(value)

  #ifndef BUTTON_UP_PIN
  #define BUTTON_UP_PIN 4
  #endif

  #ifndef BUTTON_CENTER_PIN
  #define BUTTON_CENTER_PIN 5
  #endif

  #ifndef BUTTON_LEFT_PIN
  #define BUTTON_LEFT_PIN 6
  #endif

  #ifndef BUTTON_DOWN_PIN
  #define BUTTON_DOWN_PIN 7
  #endif

  #ifndef BUTTON_RIGHT_PIN
  #define BUTTON_RIGHT_PIN 8
  #endif

  const int UP = BUTTON_UP_PIN;
  const int CENTER = BUTTON_CENTER_PIN;
  const int LEFT = BUTTON_LEFT_PIN;
  const int DOWN = BUTTON_DOWN_PIN;
  const int RIGHT = BUTTON_RIGHT_PIN;

  const int CS_PIN = 10;
  const int RST_PIN = 14;
  const int DC_PIN = 15;

  const unsigned long DEBOUNCE_MS = 30;
  const unsigned long TICK_MS = DECAY_INTERVAL_MS;
  const int8_t DECAY_RATE = 1;  
  const int EXP_AMT = EXP_AMOUNT;

  const char* name = STRINGIFY(PET_NAME);

  struct Button {
    uint8_t pin;
    bool lastReading = HIGH;
    bool stableReading = HIGH;
    volatile bool changed = false;
    unsigned long lastChanged = 0;
  };

  struct TamagotchiState 
  {
    int8_t stage;
    int8_t exp;
    int8_t happiness;
    int8_t hunger;
    int8_t health;
  };

  Button buttons[] = {
    {UP},
    {CENTER},
    {LEFT},
    {DOWN},
    {RIGHT}
  };

  void aiTask(void* arg) {
    PromptMessage message;

    for (;;) {
      // Sleep until a prompt arrives.
      if (xQueueReceive(promptQueue, &message, portMAX_DELAY) == pdTRUE) {
        Serial.print("\n[completion]: ");
        Serial.println(message.text);

        engine.generate(
          message.text,
          [](const char* text) {
            Serial.print(text);
            vTaskDelay(1);
          },
          100,
          0.8f
        );

        Serial.println("\n[AI finished]");
      }
    }
  }

  void readPrompt() {
    static PromptMessage input{};
    static size_t length = 0;
    static bool tooLong = false;

    while (Serial.available()) {
      char c = Serial.read();

      if (c == '\r') continue;

      if (c == '\n') {
        if (tooLong) {
          Serial.println("Prompt too long; discarded.");
        } else if (length > 0 && promptQueue != nullptr) {
          input.text[length] = '\0';

          if (xQueueSend(promptQueue, &input, 0) != pdTRUE) {
            Serial.println("Queue full; try again later.");
          }
        }

        length = 0;
        tooLong = false;
      } else if (length < sizeof(input.text) - 1) {
        input.text[length++] = c;
      } else {
        tooLong = true;
      }
    }
  }

  void ARDUINO_ISR_ATTR buttonIsr(void* arg) {
    Button *button = static_cast<Button *>(arg);
    button->changed = true;
  }

  Adafruit_ST7789 tft(CS_PIN, DC_PIN, RST_PIN);

  Preferences prefs;

  TamagotchiState currentState{BABY, 0, START_HAPPINESS, START_HUNGER, START_HEALTH};

  unsigned long lastTick = 0;

  void drawScreen() {
    tft.fillScreen(ST77XX_BLACK);
    tft.setTextColor(ST77XX_WHITE);
    tft.setTextSize(2);
    tft.setCursor(0, 10);
    tft.print(name);
    tft.println(" says hi!");

    tft.print("Stage: ");
    switch (currentState.stage) {
      case BABY:  tft.println("BABY"); break;
      case KID:   tft.println("KID"); break;
      case ADULT: tft.println("ADULT"); break;
    }

    tft.print("EXP: ");
    tft.println((int)currentState.exp);

    tft.print("Happy: ");
    tft.println((int)currentState.happiness);

    tft.print("Hunger: ");
    tft.println((int)currentState.hunger);

    tft.print("Health: ");
    tft.println((int)currentState.health);

    drawAvatar(currentState.stage);
  }

  uint16_t brown = tft.color565(160, 130, 110);
  uint16_t pink = tft.color565(225, 170, 200);

void drawHead(int16_t x, int16_t y, int16_t radius) {
  tft.fillCircle(x, y, radius, brown);

  // Eyes
  tft.fillCircle(x - 7, y - 5, 3, ST77XX_BLACK);
  tft.fillCircle(x + 7, y - 5, 3, ST77XX_BLACK);

  // Nose
  tft.fillCircle(x, y + 5, 4, pink);
}

void drawAvatar(int8_t stage) {
  switch (stage) {
    case BABY:
      drawHead(120, 175, 30);
      break;

    case KID:
      drawHead(96, 180, 24);
      drawHead(144, 180, 24);
      break;

    case ADULT:
      // Draw the upper head first so the front heads overlap it.
      drawHead(120, 157, 24);
      drawHead(96, 192, 24);
      drawHead(144, 192, 24);
      break;
  }
}

  void resetTamagotchi() {
    currentState = TamagotchiState{BABY, 0, START_HAPPINESS, START_HUNGER, START_HEALTH};
    prefs.putBytes("state", &currentState, sizeof(currentState));
  }

  void enableSleep() {
    prefs.putBytes("state", &currentState, sizeof(currentState));
    do {
      while (digitalRead(DOWN) == LOW) {
        delay(1);
      }
      delay(DEBOUNCE_MS);
    } while (digitalRead(DOWN) == LOW);
    
    Serial.println("Going to sleep");
    Serial.flush();

    //gpt code that keeps GPIO 7 as pullup even during sleep
    rtc_gpio_pullup_en((gpio_num_t)DOWN);
    rtc_gpio_pulldown_dis((gpio_num_t)DOWN);
    esp_sleep_pd_config(ESP_PD_DOMAIN_RTC_PERIPH, ESP_PD_OPTION_ON);
    esp_deep_sleep_start();
  }

  void handleAction(uint8_t direction) {
    if(direction == CENTER) {
      resetTamagotchi();
    } else if (direction == DOWN) {
      enableSleep();
    } else {  
      switch(direction) {
        case RIGHT: // play
          currentState.happiness += currentState.happiness < 100 ? 1 : 0;
          break;
        case LEFT: // heal
          currentState.health += currentState.health < 100 ? 1 : 0;
          break;
        case UP: // feed
          currentState.hunger += currentState.hunger > 0 ? -1 : 0;
          break;
      }
      addExp(EXP_AMT);
      prefs.putBytes("state", &currentState, sizeof(currentState));
    }
  }

  void addExp(uint8_t amount) {
    if(currentState.exp < 100) {
      currentState.exp += amount;
    }
  }

  void setup() {
    //gpt code that waits until sleep wakeup is released before starting
    rtc_gpio_deinit((gpio_num_t)DOWN);
    pinMode(DOWN, INPUT_PULLUP);

    // Wait until the wake button has been released.
    do {
      while (digitalRead(DOWN) == LOW) {
        delay(1);
      }
      delay(DEBOUNCE_MS);
    } while (digitalRead(DOWN) == LOW);

    Serial.begin(115200);

    // Button attaching
    for(Button &button : buttons) {
      pinMode(button.pin, INPUT_PULLUP);
      attachInterruptArg(button.pin, buttonIsr, &button, CHANGE);
    }

    //sleep wakeup
    esp_sleep_enable_ext1_wakeup(1ULL << DOWN, ESP_EXT1_WAKEUP_ANY_LOW);

    // Tamagotchi state management
    prefs.begin("Tamagotchi", false);

    if(!prefs.isKey("state")) {
      prefs.putBytes("state", &currentState, sizeof(currentState)); // if doesn't exist, store default state
    } else {
      prefs.getBytes("state", &currentState, sizeof(currentState)); // if exists, get stored state
    }

    tft.init(240, 240);
    tft.setRotation(0);

    lastTick = millis();
    drawScreen();

    if (!LittleFS.begin(false)) {
      Serial.println("FAIL: LittleFS mount");
      return;
    }

    if (!engine.init(LittleFS, "/stories2M_q8.bin", "/tok512.bin")) {
      Serial.println("FAIL: AI initialization");
      return;
    }

    promptQueue = xQueueCreate(2, sizeof(PromptMessage));
    configASSERT(promptQueue != nullptr);

    BaseType_t result = xTaskCreatePinnedToCore(
      aiTask,
      "AI",
      16384,
      nullptr,
      1,
      nullptr,
      1
    );

    configASSERT(result == pdPASS);

    Serial.println("AI task ready. Send a prompt using Newline.");
  }

  void tick() {
      // check evolve
      if(currentState.exp >= 100 && currentState.stage < ADULT) {
        currentState.stage++;
        currentState.exp = 0;
      } 

      currentState.happiness -= DECAY_RATE;
      if(currentState.happiness < 0) currentState.happiness = 0;
      currentState.health -= DECAY_RATE;
      if(currentState.health < 0) currentState.health = 0;
      currentState.hunger += DECAY_RATE;
      if(currentState.hunger > 100) currentState.hunger = 100;

      prefs.putBytes("state", &currentState, sizeof(currentState));
  }


  static unsigned long lastBlink = 0;
  static bool ledOn = false;
  void loop() {
    unsigned long now = millis();

    if(now - lastTick >= TICK_MS) {
      tick();
      lastTick = now;
      drawScreen();
    }

    for(int i = 0; i < 5; i++) {
      Button &button = buttons[i];
      bool changed = button.changed;
      button.changed = false;

      bool reading = digitalRead(button.pin);
      
      if(changed || reading != button.lastReading) {
        button.lastReading = reading;
        button.lastChanged = now;
      }

      if(reading != button.stableReading && now - button.lastChanged >= DEBOUNCE_MS) {
        button.stableReading = reading;
        if(reading == LOW) {
          handleAction(button.pin);
          drawScreen();
        }
      }
    }


    if (millis() - lastBlink >= 500) {
      lastBlink = millis();
      ledOn = !ledOn;

      rgbLedWrite(BOARD_LED_PIN, 0, ledOn ? 20 : 0, 0);
    }

    readPrompt();
    delay(1);
  }

