/**
 * Copyright Jerily LTD. All Rights Reserved.
 * SPDX-FileCopyrightText: 2023 Neofytos Dimitriou (neo@jerily.cy)
 * SPDX-License-Identifier: MIT.
 */

#ifndef GGML_TCL_CGRAPH_H
#define GGML_TCL_CGRAPH_H

#include "common.h"

GGML_TCL_CMD(ml_NewGraphCmd);
GGML_TCL_CMD(ml_NewGraphCustomCmd);
GGML_TCL_CMD(ml_GraphComputeCmd);
GGML_TCL_CMD(ml_GraphResetCmd);
GGML_TCL_CMD(ml_GraphDumpDotCmd);
GGML_TCL_CMD(ml_BuildForwardExpandCmd);
GGML_TCL_CMD(ml_BuildBackwardExpandCmd);
GGML_TCL_CMD(ml_GraphCpyCmd);

#endif //GGML_TCL_CGRAPH_H
