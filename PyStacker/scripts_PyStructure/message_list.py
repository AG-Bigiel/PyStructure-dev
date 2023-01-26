#additonal deprication warning:
class bcolors:
    """
    Define colors for warning and fail
    """
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    

messages =[
    ' "SPEC_TPEAK" and "SPEC_RMS" are depricated and will be removed in a future update. Use "INT_TPEAK" and "INT_RMS" instead.',
]
def print_warning(i):
    print(f'{bcolors.WARNING}[Attention]'+ messages[i]+f'{bcolors.ENDC}')


