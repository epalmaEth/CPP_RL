#include <iostream>

#include "scripts/play.h"
#include "scripts/train.h"
#include "utils/types.h"

int main(int argc, char* argv[]) {
    bool playing;
    if (string(argv[1]) == "train") {
        playing = false;
    }else if (string(argv[1]) == "play") {
        playing = true;
    }else {
        std::cerr << "Invalid argument: " << argv[1] << ". Use train or play" << std::endl;
        return 1;
    }

    if (playing) {
        return play(string(argv[2]));
    } else {
        return train(string(argv[2]));
    }
}
