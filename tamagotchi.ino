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
  #include "freertos/semphr.h"
  #include <atomic>
  #include <cstring>
  #include "esp_adc/adc_continuous.h"
  #include "esp_err.h"
  #include "soc/soc_caps.h"


  Llama2Engine engine;

  struct PromptMessage {
    char text[192];
  };

  QueueHandle_t promptQueue = nullptr;

  QueueHandle_t beepQueue = nullptr;

  SemaphoreHandle_t buttonSemaphore = nullptr;
  portMUX_TYPE buttonMux = portMUX_INITIALIZER_UNLOCKED;

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

  QueueHandle_t displayQueue = nullptr;

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

    constexpr uint8_t POT_ADC_PIN = 1;
    constexpr uint8_t SPEAKER_PIN = 2;

    // Enable only after confirming the audio wiring.
    constexpr bool AUDIO_ENABLED = true;

    std::atomic<int> potRaw{0};
    std::atomic<int> beepFrequency{500};  

    constexpr uint32_t ADC_FRAME_BYTES = 256;
    constexpr uint32_t ADC_SAMPLE_HZ = 20000;
    static adc_continuous_handle_t adcHandle = nullptr;
    static adc_channel_t potChannel;

    // Driver callback executes in interrupt context: only wake the task.
    static bool IRAM_ATTR adcDone(adc_continuous_handle_t handle,
                                const adc_continuous_evt_data_t* event,
                                void* userData) {
        BaseType_t higherPriorityTaskWoken = pdFALSE;
        vTaskNotifyGiveFromISR(static_cast<TaskHandle_t>(userData),
                              &higherPriorityTaskWoken);
        return higherPriorityTaskWoken == pdTRUE;
    }

    static void configureAdc() {
        adc_unit_t unit;
        ESP_ERROR_CHECK(adc_continuous_io_to_channel(POT_ADC_PIN,
                                                  &unit, &potChannel));
        if (unit != ADC_UNIT_1) {
            Serial.println("The potentiometer must use ADC1.");
            abort();
        }

        adc_continuous_handle_cfg_t handleConfig = {};
        handleConfig.max_store_buf_size = 8192;
        handleConfig.conv_frame_size = ADC_FRAME_BYTES;
        ESP_ERROR_CHECK(adc_continuous_new_handle(&handleConfig, &adcHandle));

        adc_digi_pattern_config_t pattern = {};
        // DB_11 is the older, widely available name for this attenuation setting.
        pattern.atten = ADC_ATTEN_DB_11;
        pattern.channel = potChannel;
        pattern.unit = ADC_UNIT_1;
        pattern.bit_width = SOC_ADC_DIGI_MAX_BITWIDTH;

        adc_continuous_config_t config = {};
        config.sample_freq_hz = ADC_SAMPLE_HZ;
        config.conv_mode = ADC_CONV_SINGLE_UNIT_1;
        config.format = ADC_DIGI_OUTPUT_FORMAT_TYPE2;
        config.pattern_num = 1;
        config.adc_pattern = &pattern;
        ESP_ERROR_CHECK(adc_continuous_config(adcHandle, &config));
    }

    void adcTask(void* arg) {
      if (AUDIO_ENABLED) {
        bool attached = ledcAttach(SPEAKER_PIN, 1000, 8);
        configASSERT(attached);
        ledcWriteTone(SPEAKER_PIN, 0);
      }

      adc_continuous_evt_cbs_t callbacks = {};
      callbacks.on_conv_done = adcDone;

      ESP_ERROR_CHECK(adc_continuous_register_event_callbacks(
        adcHandle, &callbacks, xTaskGetCurrentTaskHandle()
      ));

      ESP_ERROR_CHECK(adc_continuous_start(adcHandle));

      uint8_t buffer[ADC_FRAME_BYTES];

      unsigned long beepStarted = 0;
      bool playing = false;
      int frequency = 500;
      int appliedFrequency = 0;

      for (;;) {
        // DMA wakes us; timeout also lets us service beep timing.
        ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(20));

        // Drain a bounded number of frames per iteration.
        for (int frame = 0; frame < 8; frame++) {
          uint32_t bytesRead = 0;

          esp_err_t err = adc_continuous_read(
            adcHandle, buffer, sizeof(buffer), &bytesRead, 0
          );

          if (err == ESP_ERR_TIMEOUT) {
            break;
          }

          ESP_ERROR_CHECK(err);

          uint32_t sum = 0;
          uint32_t count = 0;

          for (uint32_t i = 0;
              i + SOC_ADC_DIGI_RESULT_BYTES <= bytesRead;
              i += SOC_ADC_DIGI_RESULT_BYTES) {
            adc_digi_output_data_t sample = {};
            memcpy(&sample, buffer + i, sizeof(sample));

            if (sample.type2.unit == ADC_UNIT_1 &&
                sample.type2.channel == potChannel) {
              sum += sample.type2.data;
              count++;
            }
          }

          if (count > 0) {
            int average = sum / count;

            frequency = map(
              constrain(average, 0, 4095),
              0, 4095,
              500, 2500
            );

            potRaw.store(average, std::memory_order_relaxed);
            beepFrequency.store(frequency, std::memory_order_relaxed);
          }
        }

        unsigned long now = millis();

        uint8_t request;

        if (xQueueReceive(beepQueue, &request, 0) == pdTRUE) {
          beepStarted = now;
          playing = true;
        }

        if (playing && now - beepStarted >= 150) {
          playing = false;
        }

        if (AUDIO_ENABLED) {
          int requestedFrequency = playing ? frequency : 0;

          if (requestedFrequency != appliedFrequency) {
            ledcWriteTone(SPEAKER_PIN, requestedFrequency);
            appliedFrequency = requestedFrequency;
          }
        }
      }
    }

  void ARDUINO_ISR_ATTR buttonIsr(void* arg) {
    Button* button = static_cast<Button*>(arg);

    portENTER_CRITICAL_ISR(&buttonMux);
    button->changed = true;
    portEXIT_CRITICAL_ISR(&buttonMux);

    BaseType_t taskWoken = pdFALSE;
    xSemaphoreGiveFromISR(buttonSemaphore, &taskWoken);

    if (taskWoken == pdTRUE) {
      portYIELD_FROM_ISR();
    }
  }

  Adafruit_ST7789 tft(CS_PIN, DC_PIN, RST_PIN);

  Preferences prefs;

  TamagotchiState currentState{BABY, 0, START_HAPPINESS, START_HUNGER, START_HEALTH};

  void drawScreen(const TamagotchiState& state) {
    tft.fillScreen(ST77XX_BLACK);
    tft.setTextColor(ST77XX_WHITE);
    tft.setTextSize(2);
    tft.setCursor(0, 10);
    tft.print(name);
    tft.println(" says hi!");

    tft.print("Stage: ");
    switch (state.stage) {
      case BABY:  tft.println("BABY"); break;
      case KID:   tft.println("KID"); break;
      case ADULT: tft.println("ADULT"); break;
    }

    tft.print("EXP: ");
    tft.println((int)state.exp);

    tft.print("Happy: ");
    tft.println((int)state.happiness);

    tft.print("Hunger: ");
    tft.println((int)state.hunger);

    tft.print("Health: ");
    tft.println((int)state.health);

    drawAvatar(state.stage);
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

  void spiTask(void* arg) {
    TamagotchiState snapshot;

    for (;;) {
      if (xQueueReceive(displayQueue, &snapshot, portMAX_DELAY) == pdTRUE) {
        drawScreen(snapshot);
      }
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
    int nextExp = (int)currentState.exp + amount;
    currentState.exp = nextExp > 100 ? 100 : nextExp;

    if (currentState.exp >= 100 && currentState.stage < ADULT) {
      currentState.stage++;
      currentState.exp = 0;
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

    buttonSemaphore = xSemaphoreCreateBinary();
    configASSERT(buttonSemaphore != nullptr);

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

    displayQueue = xQueueCreate(1, sizeof(TamagotchiState));
    configASSERT(displayQueue != nullptr);

    tft.init(240, 240);
    tft.setRotation(0);

    xQueueOverwrite(displayQueue, &currentState);

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

    BaseType_t spiResult = xTaskCreatePinnedToCore(
      spiTask,
      "SPI",
      4096,
      nullptr,
      1,
      nullptr,
      0
    );

    configASSERT(spiResult == pdPASS);

    beepQueue = xQueueCreate(1, sizeof(uint8_t));
    configASSERT(beepQueue != nullptr);
    
    BaseType_t statsResult = xTaskCreatePinnedToCore(
      statsTask,
      "Stats",
      8192,
      nullptr,
      2,
      nullptr,
      1
    );

    configASSERT(statsResult == pdPASS);

    Serial.println("Stats task ready.");

    configureAdc();

    BaseType_t adcResult = xTaskCreatePinnedToCore(
      adcTask,
      "ADC Audio",
      4096,
      nullptr,
      1,
      nullptr,
      0
    );

    configASSERT(adcResult == pdPASS);
  }

  void tick() {
    bool changed = false;

    if (currentState.happiness > 0) {
      currentState.happiness--;
      changed = true;
    }

    if (currentState.health > 0) {
      currentState.health--;
      changed = true;
    }

    if (currentState.hunger < 100) {
      currentState.hunger++;
      changed = true;
    }

    prefs.putBytes("state", &currentState, sizeof(currentState));

    if (changed) {
      uint8_t request = 1;
      xQueueOverwrite(beepQueue, &request);
    }
  }

  void statsTask(void* arg) {
    unsigned long lastDecay = millis();

    for (;;) {
      // Wake on button activity or every 10 ms.
      // The timeout also lets debounce and decay continue.
      xSemaphoreTake(buttonSemaphore, pdMS_TO_TICKS(10));

      bool redraw = false;

      for (Button& button : buttons) {
        portENTER_CRITICAL(&buttonMux);
        bool changed = button.changed;
        button.changed = false;
        portEXIT_CRITICAL(&buttonMux);

        bool reading = digitalRead(button.pin);
        unsigned long now = millis();

        if (changed || reading != button.lastReading) {
          button.lastReading = reading;
          button.lastChanged = now;
        }

        if (reading != button.stableReading &&
            now - button.lastChanged >= DEBOUNCE_MS) {
          button.stableReading = reading;

          if (reading == LOW) {
            handleAction(button.pin);
            redraw = true;
          }
        }
      }

      unsigned long now = millis();

      if (now - lastDecay >= TICK_MS) {
        lastDecay = now;
        tick();
        redraw = true;
      }

      if (redraw) {
        xQueueOverwrite(displayQueue, &currentState);
      }
    }
  }


  static unsigned long lastBlink = 0;
  static bool ledOn = false;
  void loop() {
    unsigned long now = millis();

    if (now - lastBlink >= 500) {
      lastBlink = now;
      ledOn = !ledOn;

      rgbLedWrite(BOARD_LED_PIN, 0, ledOn ? 20 : 0, 0);
    }

    readPrompt();
    delay(1);
  }

