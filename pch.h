#pragma once

#define _USE_MATH_DEFINES

#include <vector>
#include <deque>
#include <curand_kernel.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "texture_types.h"
#include <chrono>
#include <array>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <optional>
#include <cmath>

#pragma warning ( push )
#pragma warning ( disable : 5105 )
#include <windows.h>
#pragma warning ( pop )
//#include <winerror.h>
#include <fileapi.h>
//#include <WinBase.h>

#include "glm/glm.hpp"
#include "glm/gtx/compatibility.hpp"
#include "glm/gtx/intersect.hpp"

#include <gsl/span>

#pragma comment (lib, "Onecore.lib")