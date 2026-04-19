Set of files to create a loop-TNR algorithms with generalized(categorical) symmetries.
The loss function is defined as $L = 0.5*||U(X,Y,Z,W) - V(P,Q,R,S)||_F^2$; where all the tensors i.e XYZWPQRS are dressed with fusion categorical/modular indices.
The gradient is then $\grad L = \frac{\partial L(X,Y,Z,W,P,Q,R,S)}{\partial Q}$ which then simplifies to $\grad L = V(P,Q,R,S) \otimes \frac{\partial V(P,Q,R,S)}{\partial Q} - U(X,Y,Z,W) \otimes \frac{\partial V(P,Q,R,S)}{\partial Q}$
Work with F. Verstraete, W. Wiesiolek et al.