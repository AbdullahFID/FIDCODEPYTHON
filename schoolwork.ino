#include <AFMotor_R4.h>
 
// Buttons (active LOW, wired to GND)
int ESTOP_PIN = 24;    // White button: Emergency-stop
int START_PIN = 22;    // Black button: Start/Pause/Stop
 
int STATUS_LED = 13;
int RELAY_PIN = 53;    // Peristaltic pump relay (Green->D53, Red->+5V, Black->GND)
 
// Turbidity sensors
int TURB_IN_PIN = A0;
int TURB_OUT_PIN = A1;
 
// Traffic light LEDs
int LED_RED = 30;
int LED_YELLOW = 32;
int LED_GREEN = 34;
 
// Motor shield setup
AF_DCMotor PRESS(1);          // M1 - filter press motor (top plate)
AF_DCMotor MIXER(2);          // M2 - mixer / dowel motor
AF_DCMotor FEED_DIRTY(4);     // M4 unchanged
AF_DCMotor FEED_CLEAN(3);     // M3 unchanged
 
 
bool mixerOn = false;
bool pressOn = false;
bool feedOn = false;
bool doseOn = false;
 
// Motor speeds
int speedMixerFast = 220;
int speedMixerSlow = 170;
int speedPress = 200;
int speedFeed = 230;
 
// Timing for each stage (in milliseconds)
// timing variables use long because millis() returns time in milliseconds
// int can only hold up to 32,767 so it would overflow in 32 seconds
// long goes up to ~2 billion so we're good for hours
long fillTime = 30000;      // 30 sec
long doseTime = 5000;       // 5 sec
long mixFastTime = 20000;   // 20 sec
long mixSlowTime = 40000;   // 40 sec
long settleTime = 120000;   // 2 min (actual)
// 5 seconds for testing
// long settleTime = 5000;   // 5 seconds
 
long sampleTime = 10000;    // 10 sec
long logInterval = 1000;    // log every second
 
long AUTO_RESET = 60000;  // 1 minute

long pressTime  = 25000;    // 25 s for press plate (gear motor)
long filterTime = 60000;    // 60 s for coag -> clean pump


// State machine states (0=WAIT_FOR_START, 1=FILL, etc.)
enum State {
  WAIT_FOR_START, // 0 
  FILL, // 30 seconds (1)
  DOSE, // 5 seconds (2)
  MIX_FAST, // 20 seconds (3)
  MIX_SLOW, // 40 seconds (4)
  SETTLE, // 120 seconds (5)
  FILTER_RUN, // 60 seconds (6)
  SAMPLE, // 10 seconds (7)
  COMPLETE // Done (8)
};

State currentState = WAIT_FOR_START;
long stateStart = 0;
long savedElapsed = 0; 
bool isPaused = false;

// Button stuff
bool clickPending = false;
long lastClickTime = 0;
long CLICK_WINDOW = 400;

long lastLog = 0;

// stateStart is a timestamp, not a constant. 
// It changes every time you call changeState() and gets set to whatever millis() returns at that 
// instant.
long getElapsed() {
  return millis() - stateStart;
}

void setLEDs(bool red, bool yellow, bool green) {
  digitalWrite(LED_RED, red ? HIGH : LOW);
  digitalWrite(LED_YELLOW, yellow ? HIGH : LOW);
  digitalWrite(LED_GREEN, green ? HIGH : LOW);
}

void stopEverything() {
  MIXER.setSpeed(0);
  MIXER.run(RELEASE);
  PRESS.setSpeed(0);
  PRESS.run(RELEASE);
  FEED_DIRTY.setSpeed(0);
  FEED_DIRTY.run(RELEASE);
  FEED_CLEAN.setSpeed(0);
  FEED_CLEAN.run(RELEASE);
  digitalWrite(RELAY_PIN, LOW);
  
  mixerOn = false;
  pressOn = false;
  feedOn = false;
  doseOn = false;
}

void startMixerFast() {
  mixerOn = true;
  MIXER.setSpeed(speedMixerFast);
  MIXER.run(FORWARD);
}

void stopMixer() {
  mixerOn = false;
  MIXER.run(RELEASE);
}

void startMixerSlow() {
  mixerOn = true;
  MIXER.setSpeed(speedMixerSlow);
  MIXER.run(FORWARD);
}

void startPress() {
  pressOn = true;
  PRESS.setSpeed(speedPress);
  PRESS.run(FORWARD);
}

void stopPress() {
  pressOn = false;
  PRESS.run(RELEASE);
}

void startFeedDirty() {
  feedOn = true;
  FEED_DIRTY.setSpeed(speedFeed);
  FEED_DIRTY.run(FORWARD);
}

void stopFeedDirty() {
  feedOn = false;
  FEED_DIRTY.run(RELEASE);
}

void startFeedClean() {
  feedOn = true;
  FEED_CLEAN.setSpeed(speedFeed);
  FEED_CLEAN.run(FORWARD);
}

void stopFeedClean() {
  feedOn = false;
  FEED_CLEAN.run(RELEASE);
}

void startDosePump() {
  doseOn = true;
  digitalWrite(RELAY_PIN, HIGH);
}

void stopDosePump() {
  doseOn = false;
  digitalWrite(RELAY_PIN, LOW);
}

void pauseCycle() {
  if (!isPaused) {
    isPaused = true;
    savedElapsed = getElapsed();  // Store position in state (e.g., 8000 ms)
    stopEverything();
    Serial.println(F("# PAUSE"));
  }
}

// Resume sets stateStart so getElapsed() returns the saved position.
// Example: Paused at 8s → resume makes getElapsed() still return 8000 ms.
void resumeCycle() {
  if (isPaused) {
    stateStart = millis() - savedElapsed;  // Set baseline to current position
    isPaused = false;
    Serial.println(F("# RESUME"));
  }
}

// when we change states we want to stop all motors/pumps first then set the new state 
void changeState(State newState) {
  stopEverything();
  currentState = newState;
  stateStart = millis();

  Serial.print(F("# STATE,"));
  if (currentState == WAIT_FOR_START) Serial.print("WAIT_FOR_START");
  else if (currentState == FILL) Serial.print("FILL");
  else if (currentState == DOSE) Serial.print("DOSE");
  else if (currentState == MIX_FAST) Serial.print("MIX_FAST");
  else if (currentState == MIX_SLOW) Serial.print("MIX_SLOW");
  else if (currentState == SETTLE) Serial.print("SETTLE");
  else if (currentState == FILTER_RUN) Serial.print("FILTER_RUN");
  else if (currentState == SAMPLE) Serial.print("SAMPLE");
  else if (currentState == COMPLETE) Serial.print("COMPLETE");
  else Serial.print("?");
  Serial.println();

  if (currentState == WAIT_FOR_START) {
    // idle: red
    setLEDs(true, false, false);
  } 
  else if (currentState == SAMPLE || currentState == COMPLETE) {
    // done: green
    setLEDs(false, false, true);
  } 
  else {
    // active: yellow
    setLEDs(false, true, false);
  }
}

// CRITICAL ADDITION I MADE A BOOL DEBOUNCER
// IT BASICALLY ENSURES WE KNOW IF THE BUTTON PRESSED IS ACTUALLY PRESSED OR IF ITS NOISE  
bool checkButton(int pin) {
  static bool lastReading = false;
  static long lastTime = 0;

  bool reading = (digitalRead(pin) == LOW);
  long now = millis();

  // did the button state change from what it was previously on
  if (reading != lastReading) {
    if ((now - lastTime) > 80) {
      lastReading = reading;
      lastTime = now;
      if (reading) {
        return true;
      }
    }
  }
  return false;
}

float getVoltage(int pin) {
  long sum = 0;
  for (int i = 0; i < 10; i++) {
    sum += analogRead(pin);
  }
  float avg = (float)sum / 10.0;
  return avg * (5.0 / 1024.0);
}

float voltageToNTU(float v) {
  return (-1120.4 * v * v) + (5742.3 * v) - 4352.9;
}

// we need a const here because if we logData with an empty string literal
// it's stored in read-only memory by the compiler. Its type is const char*.

void logData(const char* tag) {
  float vIn = getVoltage(TURB_IN_PIN);
  float vOut = getVoltage(TURB_OUT_PIN);
  float ntuIn = voltageToNTU(vIn);
  float ntuOut = voltageToNTU(vOut);

  Serial.print(millis());
  Serial.print(',');
  if (currentState == WAIT_FOR_START) Serial.print("WAIT_FOR_START");
  else if (currentState == FILL) Serial.print("FILL");
  else if (currentState == DOSE) Serial.print("DOSE");
  else if (currentState == MIX_FAST) Serial.print("MIX_FAST");
  else if (currentState == MIX_SLOW) Serial.print("MIX_SLOW");
  else if (currentState == SETTLE) Serial.print("SETTLE");
  else if (currentState == FILTER_RUN) Serial.print("FILTER_RUN");
  else if (currentState == SAMPLE) Serial.print("SAMPLE");
  else if (currentState == COMPLETE) Serial.print("COMPLETE");
  else Serial.print("?");
  Serial.print(',');
  Serial.print(tag);
  Serial.print(',');
  Serial.print(vIn, 3);
  Serial.print(',');
  Serial.print(ntuIn, 1);
  Serial.print(',');
  Serial.print(vOut, 3);
  Serial.print(',');
  Serial.print(ntuOut, 1);
  Serial.print(',');
  Serial.print(mixerOn ? 1 : 0);
  Serial.print(',');
  Serial.print(pressOn ? 1 : 0);
  Serial.print(',');
  Serial.print(feedOn ? 1 : 0);
  Serial.print(',');
  Serial.println(doseOn ? 1 : 0);
}

bool checkEstop() {
  // without static the variables reset to make zero
  static long lowStart = 0;
  static int lastVal = HIGH;

  int val = digitalRead(ESTOP_PIN);
  long now = millis();
  if (val != lastVal) {
    lastVal = val;
    Serial.print(F("# ESTOP_RAW,"));
    Serial.print(now);
    Serial.print(',');
    Serial.println(val == LOW ? F("LOW") : F("HIGH"));
  }

  if (val == LOW) {
    if (lowStart == 0) { // First time seeing it pressed?
      lowStart = now;  // Remember when it started
    }
    if (now - lowStart >= 100) {
      return true;
    }
  } else {
    lowStart = 0;  // Button is not pressed teset the timer
  }

  return false;
}

void emergencyStop() {
  stopEverything();
  Serial.println(F("=== EMERGENCY STOP ==="));
  Serial.println(F("System locked"));

  while (true) {
    setLEDs(true, true, true);
    digitalWrite(STATUS_LED, HIGH);
    delay(400);
    setLEDs(false, false, false);
    digitalWrite(STATUS_LED, LOW);
    delay(400);
  }
}

void handleClick() {
  if ((currentState == WAIT_FOR_START) || (currentState == COMPLETE)) {
    if (!isPaused) {
      Serial.println(F("# START"));
      changeState(FILL);
    }
  } else {
    if (isPaused) {
      resumeCycle();
    } else {
      pauseCycle();
    }
  }
}

void handleDoubleClick() {
  isPaused = false;
  stopEverything();
  Serial.println(F("# STOP"));
  changeState(WAIT_FOR_START);
}

void runStateMachine() {
  long elapsed = getElapsed();
  
  if (currentState == WAIT_FOR_START) {
    stopEverything();
  }
  
  else if (currentState == FILL) {
    startFeedDirty();
    if (elapsed >= fillTime) {
      changeState(DOSE);
    }
  }
  
  else if (currentState == DOSE) {
    stopFeedDirty();
    startDosePump();
    if (elapsed >= doseTime) {
      changeState(MIX_FAST);
    }
  }
  
  else if (currentState == MIX_FAST) {
    stopDosePump();
    startMixerFast();
    if (elapsed >= mixFastTime) {
      changeState(MIX_SLOW);
    }
  }
  
  else if (currentState == MIX_SLOW) {
    startMixerSlow();
    if (elapsed >= mixSlowTime) {
      changeState(SETTLE);
    }
  }
  
  else if (currentState == SETTLE) {
    stopMixer();
    
    static long lastBlink = 0;
    static bool blinkOn = false;
    long now = millis();
    if (now - lastBlink >= 500) {
      lastBlink = now;
      blinkOn = !blinkOn;
      setLEDs(false, blinkOn, false);
    }

    if (elapsed >= settleTime) {
      changeState(FILTER_RUN);
    }
  }
  
  else if (currentState == FILTER_RUN) {
  long elapsed = getElapsed();

  // --- PRESS PLATE: only first pressTime ms ---
  if (elapsed < pressTime) {
    if (!pressOn) {
      startPress();
    }
  } else {
    if (pressOn) {
      stopPress();
    }
  }

  // --- COAG -> CLEAN PUMP: full filterTime ms ---
  if (elapsed < filterTime) {
    if (!feedOn) {
      startFeedClean();
    }
  } else {
    if (feedOn) {
      stopFeedClean();
    }
    // when pump time is done, move on to SAMPLE
    changeState(SAMPLE);
  }

  // --- Blinking green LED while filtering ---
  static long lastBlink = 0;
  static bool blinkOn = false;
  long now = millis();

  if (now - lastBlink >= 300) {
    lastBlink = now;
    blinkOn = !blinkOn;
  }

  setLEDs(false, false, blinkOn);
}

  
  else if (currentState == SAMPLE) {
    stopEverything();
    if (elapsed >= sampleTime) {
      changeState(COMPLETE);
    }
  }
  
  else if (currentState == COMPLETE) {
    stopEverything();
    if (getElapsed() >= AUTO_RESET) {
      Serial.println(F("# AUTO-RESET"));
      isPaused = false;
      changeState(WAIT_FOR_START);
    }
  }
}

void setup() {
  Serial.begin(9600);
  delay(50);

  pinMode(STATUS_LED, OUTPUT);
  pinMode(ESTOP_PIN, INPUT_PULLUP);
  pinMode(START_PIN, INPUT_PULLUP);
  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, LOW);

  pinMode(LED_RED, OUTPUT);
  pinMode(LED_YELLOW, OUTPUT);
  pinMode(LED_GREEN, OUTPUT);

  stopEverything();

  Serial.println(F("time_ms,state,tag,V_in,NTU_in,V_out,NTU_out,mixer,press,feed,dose"));
  changeState(WAIT_FOR_START);

  Serial.println(F("# Ready"));
  Serial.println(F("# Single-click: pause/resume"));
  Serial.println(F("# Double-click: stop"));
  Serial.println(F("# E-STOP: emergency shutdown"));
}

void loop() {
  if (checkEstop()) {
    emergencyStop();
  }

  long now = millis();

  // this stores just to memory (except for double click)
  if (checkButton(START_PIN)) {  // Button was just pressed
    if (!clickPending) {  // This is the FIRST click
      clickPending = true;  // Flag that we're waiting to see if there's a 2nd click
      lastClickTime = now;  // Remember when this first click happened
    } else {  // This is a SECOND click (clickPending is already true)
      if (now - lastClickTime <= CLICK_WINDOW) {  // Was it within 400ms of the first click?
        clickPending = false;  // Clear the flag
        handleDoubleClick();  // YES - it's a double-click! Stop the cycle
      } else {  // More than 400ms passed since first click
        lastClickTime = now;  // Too slow, treat THIS as a new first click instead
      }
    }
  }

  // If we had one click and 400ms passed without a second click
  if (clickPending && (now - lastClickTime > CLICK_WINDOW)) {
    clickPending = false;  // Clear the flag
    handleClick();  // It was just a single-click, pause/resume
  }

  if (!isPaused) {
    runStateMachine();
  }

  // log data at a regaular interval so we dont spam the serial monitor
  if (millis() - lastLog >= logInterval) {
    lastLog = millis();
    if (!isPaused && currentState != WAIT_FOR_START) {
      logData("");
    }
  }

  delay(5); // bounce so it doesnt loop really quick gives the CPU a breather
}