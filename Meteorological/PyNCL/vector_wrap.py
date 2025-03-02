import numpy as np
import numpy.ma as ma
from typing import List, Dict, Tuple, Union, Any, Optional


class ResInfo:
    """Resource information class equivalent to C struct"""

    def __init__(self):
        self.id = 0
        self.strings = []
        self.nstrings = 0


class nglRes:
    """Class to hold resource list"""

    def __init__(self):
        pass


class nglPlotId:
    """Class to hold plot IDs"""

    def __init__(self):
        self.base = []
        self.nbase = 0
        self.contour = []
        self.ncontour = 0
        self.vector = []
        self.nvector = 0
        self.streamline = []
        self.nstreamline = 0
        self.map = []
        self.nmap = 0
        self.xy = []
        self.nxy = 0
        self.xydspec = []
        self.nxydspec = 0
        self.text = []
        self.ntext = 0
        self.primitive = []
        self.nprimitive = 0
        self.labelbar = []
        self.nlabelbar = 0
        self.legend = []
        self.nlegend = 0
        self.cafield = []
        self.ncafield = 0
        self.sffield = []
        self.nsffield = 0
        self.vffield = []
        self.nvffield = 0


def NhlRLCreate(mode):
    """Create a resource list with the specified mode"""
    return mode


def NhlRLClear(rlist):
    """Clear a resource list"""
    pass


def NhlRLSetStringArray(rlist, key, strings, size):
    """Set a string array in the resource list"""
    pass


def NhlRLSetDoubleArray(rlist, key, values, size):
    """Set a double array in the resource list"""
    pass


def NhlRLSetIntegerArray(rlist, key, values, size):
    """Set an integer array in the resource list"""
    pass


def NhlRLSetMDLongArray(rlist, key, values, ndims, dims):
    """Set a multi-dimensional long array in the resource list"""
    pass


def NhlRLSetMDDoubleArray(rlist, key, values, ndims, dims):
    """Set a multi-dimensional double array in the resource list"""
    pass


def NhlRLSetInteger(rlist, key, value):
    """Set an integer in the resource list"""
    pass


def NhlRLSetDouble(rlist, key, value):
    """Set a double in the resource list"""
    pass


def NhlRLSetString(rlist, key, value):
    """Set a string in the resource list"""
    pass


def NhlSETRL():
    """Resource list mode constant"""
    return 1


def is_string_type(obj):
    """Check if an object is a string"""
    return isinstance(obj, str)


def as_utf8_char(obj):
    """Convert object to UTF-8 string"""
    if isinstance(obj, bytes):
        return obj.decode('utf-8')
    return str(obj)


def vector_wrap(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11,
                arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21):
    """
    Actual implementation of vector_wrap function
    This would be replaced with the actual functionality
    """
    # Placeholder implementation
    result = nglPlotId()
    # Set some sample values
    result.base = [1, 2, 3]
    result.nbase = len(result.base)
    result.vector = [4, 5, 6]
    result.nvector = len(result.vector)
    # Add other properties as needed
    return result


def _wrap_vector_wrap(self, *args):
    """
    Python wrapper for vector_wrap function

    This function handles the conversion of Python objects to C types and vice versa
    """
    if len(args) != 21:
        raise ValueError("vector_wrap requires 21 arguments")

    # Extract arguments
    obj0, obj1, obj2, obj3, obj4, obj5, obj6, obj7, obj8, obj9, obj10, obj11, obj12, obj13, obj14, obj15, obj16, obj17, obj18, obj19, obj20 = args

    # Convert arguments to appropriate types
    arg1 = int(obj0)

    # Convert numpy arrays
    arr1 = np.ascontiguousarray(obj1, dtype=np.float64)
    arg2 = arr1.data

    arr2 = np.ascontiguousarray(obj2, dtype=np.float64)
    arg3 = arr2.data

    # Convert strings
    arg4 = as_utf8_char(obj3)
    arg5 = as_utf8_char(obj4)

    # Convert integers
    arg6 = int(obj5)
    arg7 = int(obj6)
    arg8 = int(obj7)

    # Convert void pointers and more strings
    arg9 = obj8
    arg10 = as_utf8_char(obj9)
    arg11 = int(obj10)
    arg12 = obj11
    arg13 = as_utf8_char(obj12)
    arg14 = int(obj13)
    arg15 = int(obj14)
    arg16 = obj15
    arg17 = obj16

    # Process dictionary resource list for arg18
    arg18 = process_resource_dict(obj17)

    # Process dictionary resource list for arg19
    arg19 = process_resource_dict(obj18)

    # Process dictionary resource list for arg20
    arg20 = process_resource_dict(obj19)

    # Use global resource list for arg21
    arg21 = nglRes()

    # Call the actual vector_wrap function
    result = vector_wrap(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9,
                         arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17,
                         arg18, arg19, arg20, arg21)

    # Convert result to Python list
    return_list = [None] * 14

    # Process base IDs
    if result.nbase == 0:
        return_list[0] = None
    else:
        l_base = [result.base[i] for i in range(result.nbase)]
        return_list[0] = l_base

    # Process contour IDs
    if result.ncontour == 0:
        return_list[1] = None
    else:
        l_contour = [result.contour[i] for i in range(result.ncontour)]
        return_list[1] = l_contour

    # Process vector IDs
    if result.nvector == 0:
        return_list[2] = None
    else:
        l_vector = [result.vector[i] for i in range(result.nvector)]
        return_list[2] = l_vector

    # Process streamline IDs
    if result.nstreamline == 0:
        return_list[3] = None
    else:
        l_streamline = [result.streamline[i] for i in range(result.nstreamline)]
        return_list[3] = l_streamline

    # Process map IDs
    if result.nmap == 0:
        return_list[4] = None
    else:
        l_map = [result.map[i] for i in range(result.nmap)]
        return_list[4] = l_map

    # Process xy IDs
    if result.nxy == 0:
        return_list[5] = None
    else:
        l_xy = [result.xy[i] for i in range(result.nxy)]
        return_list[5] = l_xy

    # Process xydspec IDs
    if result.nxydspec == 0:
        return_list[6] = None
    else:
        l_xydspec = [result.xydspec[i] for i in range(result.nxydspec)]
        return_list[6] = l_xydspec

    # Process text IDs
    if result.ntext == 0:
        return_list[7] = None
    else:
        l_text = [result.text[i] for i in range(result.ntext)]
        return_list[7] = l_text

    # Process primitive IDs
    if result.nprimitive == 0:
        return_list[8] = None
    else:
        l_primitive = [result.primitive[i] for i in range(result.nprimitive)]
        return_list[8] = l_primitive

    # Process labelbar IDs
    if result.nlabelbar == 0:
        return_list[9] = None
    else:
        l_labelbar = [result.labelbar[i] for i in range(result.nlabelbar)]
        return_list[9] = l_labelbar

    # Process legend IDs
    if result.nlegend == 0:
        return_list[10] = None
    else:
        l_legend = [result.legend[i] for i in range(result.nlegend)]
        return_list[10] = l_legend

    # Process cafield IDs
    if result.ncafield == 0:
        return_list[11] = None
    else:
        l_cafield = [result.cafield[i] for i in range(result.ncafield)]
        return_list[11] = l_cafield

    # Process sffield IDs
    if result.nsffield == 0:
        return_list[12] = None
    else:
        l_sffield = [result.sffield[i] for i in range(result.nsffield)]
        return_list[12] = l_sffield

    # Process vffield IDs
    if result.nvffield == 0:
        return_list[13] = None
    else:
        l_vffield = [result.vffield[i] for i in range(result.nvffield)]
        return_list[13] = l_vffield

    return return_list


def process_resource_dict(obj_dict):
    """
    Process a dictionary of resources and create a ResInfo object

    Args:
        obj_dict: Dictionary containing resource key-value pairs

    Returns:
        ResInfo object with processed resources
    """
    if not isinstance(obj_dict, dict):
        print("Resource lists must be dictionaries")
        return None

    # Create resource list
    rlist = NhlRLCreate(NhlSETRL())
    NhlRLClear(rlist)

    # Create ResInfo object
    trname = ResInfo()
    trname.nstrings = len(obj_dict)
    trnames = [None] * trname.nstrings

    # Process each key-value pair
    count = 0
    for key, value in obj_dict.items():
        trnames[count] = as_utf8_char(key)
        count += 1

        # Process tuple values
        if isinstance(value, tuple):
            # Check for nested lists or tuples
            if any(isinstance(item, (list, tuple)) for item in value):
                print("Tuple values are not allowed to have list or tuple items.")
                return None

            list_len = len(value)

            # Determine type of tuple elements
            if is_string_type(value[0]):
                # Check all elements are strings
                if not all(is_string_type(item) for item in value):
                    print(f"All items in the tuple value for resource {key} must be strings")
                    return None

                # Process string tuple
                strings = [as_utf8_char(item) for item in value]
                NhlRLSetStringArray(rlist, as_utf8_char(key), strings, list_len)
            else:
                # Check all elements are ints or floats
                if not all(isinstance(item, (int, float)) for item in value):
                    print(f"All items in the tuple value for resource {key} must be ints or floats.")
                    return None

                # Check if any element is float
                if any(isinstance(item, float) for item in value):
                    # Process float tuple
                    dvals = [float(item) for item in value]
                    NhlRLSetDoubleArray(rlist, as_utf8_char(key), dvals, list_len)
                else:
                    # Process int tuple
                    ivals = [int(item) for item in value]
                    NhlRLSetIntegerArray(rlist, as_utf8_char(key), ivals, list_len)

        # Process list values
        elif isinstance(value, list):
            # Check for nested lists or tuples
            if any(isinstance(item, (list, tuple)) for item in value):
                print("Use NumPy arrays for multiple dimension arrays.")
                return None

            list_len = len(value)

            # Determine type of list elements
            if is_string_type(value[0]):
                # Check all elements are strings
                if not all(is_string_type(item) for item in value):
                    print(f"All items in the list value for resource {key} must be strings")
                    return None

                # Process string list
                strings = [as_utf8_char(item) for item in value]
                NhlRLSetStringArray(rlist, as_utf8_char(key), strings, list_len)
            else:
                # Check all elements are ints or floats
                if not all(isinstance(item, (int, float)) for item in value):
                    print(f"All items in the list value for resource {key} must be ints or floats.")
                    return None

                # Check if any element is float
                if any(isinstance(item, float) for item in value):
                    # Process float list
                    dvals = [float(item) for item in value]
                    NhlRLSetDoubleArray(rlist, as_utf8_char(key), dvals, list_len)
                else:
                    # Process int list
                    ivals = [int(item) for item in value]
                    NhlRLSetIntegerArray(rlist, as_utf8_char(key), ivals, list_len)

        # Process scalar values
        elif np.isscalar(value):
            # Python scalar types
            if isinstance(value, int):
                NhlRLSetInteger(rlist, as_utf8_char(key), int(value))
            elif isinstance(value, float):
                NhlRLSetDouble(rlist, as_utf8_char(key), float(value))
            elif is_string_type(value):
                NhlRLSetString(rlist, as_utf8_char(key), as_utf8_char(value))

        # Process numpy array values
        elif isinstance(value, np.ndarray):
            array_type = value.dtype

            # Process different array types
            if np.issubdtype(array_type, np.integer):
                arr = np.ascontiguousarray(value, dtype=np.int64)
                lvals = arr.flatten()
                ndims = arr.ndim
                len_dims = arr.shape

                NhlRLSetMDLongArray(rlist, as_utf8_char(key), lvals, ndims, len_dims)
            elif np.issubdtype(array_type, np.floating):
                arr = np.ascontiguousarray(value, dtype=np.float64)
                dvals = arr.flatten()
                ndims = arr.ndim
                len_dims = arr.shape

                NhlRLSetMDDoubleArray(rlist, as_utf8_char(key), dvals, ndims, len_dims)
            else:
                print("NumPy arrays must be of type int, int32, float, float32, or float64.")
                return None
        else:
            print(f"Value for keyword {key} is invalid.")
            return None

    trname.strings = trnames
    trname.id = rlist

    return trname


# Make this function available as a method
if __name__ == "__main__":
    # Example usage
    class ExampleClass:
        def vector_wrap(self, *args):
            return _wrap_vector_wrap(self, *args)


    example = ExampleClass()

    # Create sample data
    arg0 = 1  # integer
    arg1 = np.array([1.0, 2.0, 3.0])  # numpy array
    arg2 = np.array([4.0, 5.0, 6.0])  # numpy array
    arg3 = "label1"  # string
    arg4 = "label2"  # string
    arg5 = 2  # integer
    arg6 = 3  # integer
    arg7 = 4  # integer
    arg8 = None  # void pointer
    arg9 = "option1"  # string
    arg10 = 5  # integer
    arg11 = None  # void pointer
    arg12 = "option2"  # string
    arg13 = 6  # integer
    arg14 = 7  # integer
    arg15 = None  # void pointer
    arg16 = None  # void pointer
    arg17 = {"color": "red", "width": 2, "labels": ["A", "B", "C"]}  # resource dict
    arg18 = {"fontsize": 12, "alignment": "center"}  # resource dict
    arg19 = {"range": [0, 100], "scale": 1.5}  # resource dict
    arg20 = None  # nglRes

    # Call the function
    result = example.vector_wrap(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8,
                                 arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16,
                                 arg17, arg18, arg19, arg20)

    print(result)