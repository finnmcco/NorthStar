#include "button-driver.h"
#include "gpio.h"
#include <gpiod.hpp>
#include <thread>

using namespace button_driver;

ButtonDriver::ButtonDriver() {
    worker = std::thread([&] () {
        while (this->running) {
            auto edge = gpio::blockUntilEdge(ButtonDriver::BUTTON_PIN, gpiod::line::edge::BOTH);
                
            if (!edge.has_value()) { continue; }
            //at this point, we know we are seeing AN edge - but we don't know if it's rising or falling
            bool val = edge.value() == gpiod::edge_event::event_type::RISING_EDGE;
            //now true == rising, false == falling
            this->buttonStatus = val;
                
                
            if (val) { //button pressed
                //if no callback has been registered, skip:
                if (!this->pressCallback.has_value()) { continue; } 
                //extract the callback and call it:
                this->pressCallback.value()();
            }
            else { //button released
                //if no callback has been registered, skip:
                if (!this->releaseCallback.has_value()) { continue; } 
                //extract the callback and call it:
                this->releaseCallback.value()(); 
            }
               
        }
    });  
}

ButtonDriver::~ButtonDriver() {
    this->running = false;
    gpio::cancelLineRequests();
    if (this->worker.joinable()) { this->worker.join(); }
}

void ButtonDriver::registerPressCallback(PressCallback callback) {
    this->pressCallback = callback;
}

void ButtonDriver::deregisterPressCallback() {
    this->pressCallback = {};
}

void ButtonDriver::registerReleaseCallback(ReleaseCallback callback) {
    this->releaseCallback = callback;
}

void ButtonDriver::deregisterReleaseCallback() {
    this->releaseCallback = {};
}

