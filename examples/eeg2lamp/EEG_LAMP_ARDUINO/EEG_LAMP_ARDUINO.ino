#include <FastLED.h>

// Change these according to your needs and LED strip type
#define LED_PIN     3
#define NUM_LEDS    80
#define LED_TYPE    WS2811
#define COLOR_ORDER GRB

#define BRIGHT_MIN  50
#define BRIGHT_MAX  255
#define FADE_DELAY  100  // ms

CRGB leds[NUM_LEDS];

// Colour gradient from turquoise to yellow
CRGB color_low  = CRGB(0, 255, 180);   // turqouise
CRGB color_high = CRGB(255, 200, 0);   // yellow

int current_brightness = BRIGHT_MIN;
int target_brightness = BRIGHT_MIN;
unsigned long last_update = 0;

void setup() {
  Serial.begin(9600);
  delay(300);

  FastLED.addLeds<LED_TYPE, LED_PIN, COLOR_ORDER>(leds, NUM_LEDS);
  FastLED.setBrightness(current_brightness);
  fill_solid(leds, NUM_LEDS, color_low);
  FastLED.show();
}

void loop() {
  // Read from serial port
  if (Serial.available()) {
    char c = Serial.read();

    if (c == '1') target_brightness = BRIGHT_MAX;
    else if (c == '0') target_brightness = BRIGHT_MIN;
  }

  // Change brightness every FADE_DELAY ms
  if (millis() - last_update >= FADE_DELAY) {
    last_update = millis();

    // Let current_brightness go towards max or min (depending on current state of target_brightness)
    if (current_brightness < target_brightness) current_brightness++;
    else if (current_brightness > target_brightness) current_brightness--;

    FastLED.setBrightness(current_brightness);

    // Interpolate lamp colour based on current brightness and possible range of brightness
    float factor = float(current_brightness - BRIGHT_MIN) / (BRIGHT_MAX - BRIGHT_MIN);
    byte r = lerp8by8(color_low.r, color_high.r, factor * 255);
    byte g = lerp8by8(color_low.g, color_high.g, factor * 255);
    byte b = lerp8by8(color_low.b, color_high.b, factor * 255);
    CRGB base_color = CRGB(r, g, b);

    // Set the colour of all LEDs
    for (int i = 0; i < NUM_LEDS; i++) {
      leds[i] = base_color;
    }

    FastLED.show();
  }
}
