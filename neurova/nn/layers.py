# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""
Base neural network module classes.

Neurova implementation and container modules.
"""

from __future__ import annotations
from typing import Iterator, Optional, Dict, Any, List
from neurova.nn.autograd import Tensor, Parameter


class Module:
    """
    Base class for all neural network modules.
    
    Neurova implementation - provides parameter management,
    training/eval modes, and forward pass interface.
    """
    
    def __init__(self):
        self._parameters: Dict[str, Parameter] = {}
        self._modules: Dict[str, Module] = {}
        self.training = True
    
    def forward(self, *args, **kwargs) -> Tensor:
        """
        Forward pass - must be implemented by subclasses.
        
        Raises
        ------
        NotImplementedError
            If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement forward()")
    
    def __call__(self, *args, **kwargs) -> Tensor:
        """Call forward pass."""
        return self.forward(*args, **kwargs)
    
    def parameters(self) -> Iterator[Parameter]:
        """
        Iterate over all parameters.
        
        Yields
        ------
        Parameter
            Each parameter in the module and its submodules
        """
        for param in self._parameters.values():
            yield param
        for module in self._modules.values():
            yield from module.parameters()
    
    def named_parameters(self, prefix: str = '') -> Iterator[tuple[str, Parameter]]:
        """
        Iterate over all parameters with names.
        
        Parameters
        ----------
        prefix : str, optional
            Prefix to add to parameter names
        
        Yields
        ------
        tuple[str, Parameter]
            Name and parameter pairs
        """
        for name, param in self._parameters.items():
            yield (prefix + name, param)
        for name, module in self._modules.items():
            subprefix = prefix + name + '.'
            yield from module.named_parameters(subprefix)
    
    def children(self) -> Iterator[Module]:
        """Iterate over immediate child modules."""
        for module in self._modules.values():
            yield module
    
    def train(self, mode: bool = True) -> Module:
        """
        Set training mode.
        
        Parameters
        ----------
        mode : bool, default=True
            Training mode if True, evaluation mode if False
        
        Returns
        -------
        Module
            Self for chaining
        """
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self
    
    def eval(self) -> Module:
        """Set evaluation mode."""
        return self.train(False)
    
    def zero_grad(self) -> None:
        """Zero out all parameter gradients."""
        for param in self.parameters():
            param.zero_grad()
    
    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        """
        Register a parameter.
        
        Parameters
        ----------
        name : str
            Parameter name
        param : Parameter or None
            Parameter to register
        """
        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, Parameter):
            raise TypeError(f"Cannot assign {type(param)} as parameter '{name}' "
                          "(Parameter or None expected)")
        else:
            self._parameters[name] = param
    
    def register_module(self, name: str, module: Optional[Module]) -> None:
        """
        Register a submodule.
        
        Parameters
        ----------
        name : str
            Module name
        module : Module or None
            Module to register
        """
        if module is None:
            self._modules[name] = None
        elif not isinstance(module, Module):
            raise TypeError(f"Cannot assign {type(module)} as child module '{name}' "
                          "(Module or None expected)")
        else:
            self._modules[name] = module
    
    def __setattr__(self, name: str, value: Any) -> None:
        """Automatically register parameters and modules."""
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        else:
            object.__setattr__(self, name, value)
    
    def __getattr__(self, name: str) -> Any:
        """Get parameters and modules."""
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_modules' in self.__dict__:
            _modules = self.__dict__['_modules']
            if name in _modules:
                return _modules[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def __repr__(self) -> str:
        lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = '\n  '.join(mod_str.split('\n'))
            lines.append(f'  ({key}): {mod_str}')
        main_str = self.__class__.__name__ + '('
        if lines:
            main_str += '\n' + '\n'.join(lines) + '\n'
        main_str += ')'
        return main_str


class Sequential(Module):
    """
    Sequential container - layers are applied in order.
    
    Neurova implementation
    
    Parameters
    ----------
    *args : Module
        Sequence of modules to chain
    
    Examples
    --------
    >>> model = Sequential(
    ...     Linear(784, 256),
    ...     ReLU(),
    ...     Linear(256, 10)
    ... )
    """
    
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self.register_module(str(idx), module)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through all modules in sequence."""
        for module in self._modules.values():
            x = module(x)
        return x
    
    def __getitem__(self, idx: int) -> Module:
        """Get module by index."""
        return self._modules[str(idx)]
    
    def __len__(self) -> int:
        """Number of modules."""
        return len(self._modules)


class ModuleList(Module):
    """
    List of modules.
    
    Neurova implementation
    
    Parameters
    ----------
    modules : list of Module, optional
        List of modules to hold
    """
    
    def __init__(self, modules: Optional[List[Module]] = None):
        super().__init__()
        if modules is not None:
            self.extend(modules)
    
    def append(self, module: Module) -> ModuleList:
        """Append a module."""
        self.register_module(str(len(self)), module)
        return self
    
    def extend(self, modules: List[Module]) -> ModuleList:
        """Extend with list of modules."""
        for module in modules:
            self.append(module)
        return self
    
    def forward(self, x: Tensor) -> Tensor:
        """Not implemented for ModuleList."""
        raise NotImplementedError("ModuleList has no forward() method")
    
    def __getitem__(self, idx: int) -> Module:
        """Get module by index."""
        return self._modules[str(idx)]
    
    def __len__(self) -> int:
        """Number of modules."""
        return len(self._modules)
    
    def __iter__(self) -> Iterator[Module]:
        """Iterate over modules."""
        return iter(self._modules.values())


class ModuleDict(Module):
    """
    Dictionary of modules.
    
    Neurova implementation
    
    Parameters
    ----------
    modules : dict of str -> Module, optional
        Dictionary of modules to hold
    """
    
    def __init__(self, modules: Optional[Dict[str, Module]] = None):
        super().__init__()
        if modules is not None:
            self.update(modules)
    
    def __setitem__(self, key: str, module: Module) -> None:
        """Set module by key."""
        self.register_module(key, module)
    
    def __getitem__(self, key: str) -> Module:
        """Get module by key."""
        return self._modules[key]
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return key in self._modules
    
    def keys(self):
        """Get keys."""
        return self._modules.keys()
    
    def values(self):
        """Get values."""
        return self._modules.values()
    
    def items(self):
        """Get items."""
        return self._modules.items()
    
    def update(self, modules: Dict[str, Module]) -> None:
        """Update with dictionary of modules."""
        for key, module in modules.items():
            self[key] = module
    
    def forward(self, x: Tensor) -> Tensor:
        """Not implemented for ModuleDict."""
        raise NotImplementedError("ModuleDict has no forward() method")
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.