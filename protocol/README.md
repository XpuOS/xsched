# Environment Variables for XSched



| Name                         | Type   | Range                                                        | Default Value | Description                                                  |
| ---------------------------- | ------ | ------------------------------------------------------------ | ------------- | ------------------------------------------------------------ |
| XSCHED_AUTO_XQUEUE           | string | ON/OFF                                                       | OFF           |                                                              |
| XSCHED_AUTO_XQUEUE_LEVEL     | int    | [1, 3]                                                       | 1             |                                                              |
| XSCHED_AUTO_XQUEUE_THRESHOLD | int    | [1, MAX_INT64]                                               | 16            |                                                              |
| ...                          |        |                                                              |               |                                                              |
| XSCHED_POLICY                | string | GLB<br />AMG<br />HPF<br />RR<br />UP<br />PUP<br />EDF<br />LAX | AMG           | GLB: managed by global scheduler<br />AMG: application managed |

