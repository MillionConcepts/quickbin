import numpy as np

from quickbin._quickbin_core import binned_std
from quickbin.tests.refalarm import RefAlarm
from quickbin.tests._quickbin_test_utils import refleak


def test_refalarm():
    """simple test of basic RefAlarm functionality"""
    x, y, z = [0, 0, 0], [0, 0, 0], [0, 0, 0]
    x[1] = z

    def pointlessly_assign_to_index_1(seq, obj):
        seq[1] = obj

    alarm = RefAlarm(verbosity="quiet")
    with alarm.context():
        pointlessly_assign_to_index_1(y, z)
    assert alarm.refcaches['default'] == [
        [{'name': 'test_refalarm', 'mismatches': {'z': 1}}]
    ]


def test_leak_detect():
    """make sure we can see an intentional C-layer reference leak."""
    thing = [1, 2, 3, 4]
    alarm = RefAlarm()
    with alarm.context():
        refleak(thing)
    assert alarm.refcaches['default'][0][0]['mismatches']['thing'] == 1


def test_bin2d_refalarm_leak():
    """straightforward check for dangling references created in the C layer."""
    iarr = np.zeros(1000, dtype='f8')
    jarr = np.zeros(1000, dtype='f8')
    varr = np.zeros(1000, dtype='f8')
    in_arrs = (iarr, jarr, varr)
    countarr = np.zeros(100, dtype='f8')
    sumarr = np.zeros(100, dtype='f8')
    meanarr = np.zeros(100, dtype='f8')
    stdarr = np.zeros(100, dtype='f8')
    out_arrs = (countarr, sumarr, meanarr, stdarr)
    ranges = (float('nan'),) * 4
    alarm = RefAlarm(verbosity="quiet")
    with alarm.context():
        binned_std(*in_arrs, *out_arrs, *ranges, 10, 10)
    assert len(alarm.refcaches['default'][0][0]['mismatches']) == 0
